#!/usr/bin/env python3
"""Debug layer 0 inputs to find where divergence starts."""

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
    # HF forward
    hf_lm = hf_model.model.language_model
    hf_hidden = hf_lm.embed_tokens(input_ids)

    # Compute PLE using HF's actual method
    per_layer_inputs_discrete = hf_lm.get_per_layer_inputs(input_ids)
    per_layer_inputs = hf_lm.project_per_layer_inputs(hf_hidden, per_layer_inputs_discrete)

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

    # FS2 forward
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    _, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
    fs2_hidden_4d = fs2_model.decoder._stack_altup(
        fs2_model.decoder_frontend.embed(input_ids) * fs2_model.decoder_frontend.scale
    )

    # Extract PLE from state_bag
    fs2_per_layer_inputs = getattr(state_bag, 'per_layer_inputs', None)

    print("="*80)
    print("Checking inputs to layer 0")
    print("="*80)

    # Compare 4D hidden states
    print("\n4D Hidden States:")
    print(f"  HF shape:  {hf_hidden_4d.shape}")
    print(f"  FS2 shape: {fs2_hidden_4d.shape}")
    diff_4d = (hf_hidden_4d - fs2_hidden_4d).abs()
    print(f"  Diff: max={diff_4d.max():.6e}, mean={diff_4d.mean():.6e}")

    # Compare PLE for layer 0
    hf_ple_layer0 = per_layer_inputs[:, :, 0, :]
    fs2_ple_layer0 = fs2_per_layer_inputs[:, :, 0, :] if fs2_per_layer_inputs is not None else None

    print("\nPLE for layer 0:")
    if fs2_ple_layer0 is not None:
        print(f"  HF shape:  {hf_ple_layer0.shape}")
        print(f"  FS2 shape: {fs2_ple_layer0.shape}")
        diff_ple = (hf_ple_layer0 - fs2_ple_layer0).abs()
        print(f"  Diff: max={diff_ple.max():.6e}, mean={diff_ple.mean():.6e}")
    else:
        print("  FS2 PLE is None!")

    # Now run layer 0 and compare outputs
    from fairseq2.models.transformer import AttentionBiasCache
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding

    attn_bias_cache = AttentionBiasCache()

    print("\n" + "="*80)
    print("Running layer 0")
    print("="*80)

    # Get position embeddings for HF
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    layer_type = hf_lm.config.layer_types[0]
    cos, sin = rope(hf_hidden_4d[0], position_ids, layer_type)

    # HF layer 0 - returns 4D corrected_predictions
    hf_layer = hf_lm.layers[0]
    hf_output_4d = hf_layer(
        hidden_states=hf_hidden_4d,
        per_layer_input=hf_ple_layer0,
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
        past_key_values=None,
        cache_position=None,
    )

    # FS2 layer 0
    fs2_layer = fs2_model.decoder.layers[0]
    fs2_output_4d = fs2_layer(
        fs2_hidden_4d, batch_layout, attn_bias_cache,
        per_layer_input=fs2_ple_layer0, state_bag=state_bag
    )

    print("\nLayer 0 output (4D):")
    print(f"  HF shape:  {hf_output_4d.shape}")
    print(f"  FS2 shape: {fs2_output_4d.shape}")
    diff_output = (hf_output_4d - fs2_output_4d).abs()
    print(f"  Diff: max={diff_output.max():.6e}, mean={diff_output.mean():.6e}")
