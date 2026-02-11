#!/usr/bin/env python3
"""Test layers 0-5 to find where divergence starts."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag

device = torch.device("cpu")

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=torch.float32, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)
input_ids = tokenizer("Hi", return_tensors="pt")["input_ids"].to(device)

with torch.no_grad():
    # HF forward through layers
    hf_lm = hf_model.model.language_model
    hf_hidden = hf_lm.embed_tokens(input_ids)

    # Compute PLE (per_layer_input)
    per_layer_embeds = hf_lm.embed_tokens_per_layer(input_ids)
    per_layer_model_projection = hf_lm.per_layer_model_projection(hf_hidden)
    per_layer_projection_scale = hf_lm.per_layer_projection_scale
    per_layer_input_raw = per_layer_embeds + per_layer_model_projection * per_layer_projection_scale
    per_layer_input_raw = per_layer_input_raw.reshape(*input_ids.shape, config.num_layers, -1)
    per_layer_input_scale = hf_lm.per_layer_input_scale
    per_layer_inputs = []
    for layer_idx in range(config.num_layers):
        ple_layer = per_layer_input_raw[:, :, layer_idx, :]
        ple_layer = hf_lm.per_layer_projection_norm(ple_layer)
        per_layer_inputs.append(ple_layer * per_layer_input_scale)

    # 4D stacking
    target_magnitude = torch.mean(hf_hidden**2, dim=-1, keepdim=True) ** 0.5
    epsilon = torch.tensor(1e-5, device=device, dtype=torch.float32)
    temp_hidden = [hf_hidden]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hf_hidden)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden.append(altup_proj)
    hf_hidden_4d = torch.stack(temp_hidden, dim=0)

    # FS2 forward through layers
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    _, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
    fs2_hidden_4d = fs2_model.decoder._stack_altup(
        fs2_model.decoder_frontend.embed(input_ids) * fs2_model.decoder_frontend.scale
    )

    # Extract PLE from state_bag (like decoder does)
    fs2_per_layer_inputs = getattr(state_bag, 'per_layer_inputs', None)

    print("Testing layers 0-5:")
    print("="*80)

    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding
    from fairseq2.models.transformer import AttentionBiasCache
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    attn_bias_cache = AttentionBiasCache()

    for layer_idx in range(min(6, config.num_layers)):
        # Get position embeddings for this layer
        layer_type = hf_lm.config.layer_types[layer_idx]
        cos, sin = rope(hf_hidden_4d[0], position_ids, layer_type)

        # HF layer
        hf_layer = hf_lm.layers[layer_idx]
        hf_hidden_4d = hf_layer(
            hidden_states=hf_hidden_4d,
            per_layer_input=per_layer_inputs[layer_idx],
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
            past_key_values=None,
            cache_position=None,
        )[0]

        # FS2 layer - extract PLE for this layer
        if fs2_per_layer_inputs is not None:
            fs2_layer_ple = fs2_per_layer_inputs[:, :, layer_idx, :]
        else:
            fs2_layer_ple = None

        fs2_layer = fs2_model.decoder.layers[layer_idx]
        fs2_hidden_4d = fs2_layer(
            fs2_hidden_4d, batch_layout, attn_bias_cache,
            per_layer_input=fs2_layer_ple, state_bag=state_bag
        )

        # Compare 4D outputs
        diff = (hf_hidden_4d - fs2_hidden_4d).abs()
        print(f"Layer {layer_idx}: max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}")

        if diff.max() > 1e-3:
            print(f"  ⚠️  Divergence detected at layer {layer_idx}!")
            break
