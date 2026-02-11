#!/usr/bin/env python3
"""Compare full model forward to find where divergence accumulates."""

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
    print("="*80)
    print("Full forward pass comparison")
    print("="*80)

    # HF full forward
    hf_output = hf_model(input_ids)
    hf_logits = hf_output.logits

    # FS2 full forward
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    fs2_logits = fs2_model(input_ids, batch_layout, state_bag=state_bag)

    print(f"\nLogits shapes: HF={hf_logits.shape}, FS2={fs2_logits.shape}")
    diff = (hf_logits - fs2_logits).abs()
    print(f"Logits diff: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Check intermediate outputs
    print("\n" + "="*80)
    print("Intermediate decoder states")
    print("="*80)

    # HF intermediate
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
    per_layer_inputs_discrete = hf_lm.get_per_layer_inputs(input_ids)
    per_layer_inputs = hf_lm.project_per_layer_inputs(hf_embeds, per_layer_inputs_discrete)

    # 4D stack
    target_magnitude = torch.mean(hf_embeds**2, dim=-1, keepdim=True) ** 0.5
    epsilon = torch.tensor(1e-5, device=device, dtype=torch.float32)
    temp_hidden = [hf_embeds]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hf_embeds)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden.append(altup_proj)
    hf_hidden_4d = torch.stack(temp_hidden, dim=0)

    # FS2 intermediate
    seqs, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
    fs2_hidden_4d = fs2_model.decoder._stack_altup(seqs)

    print(f"\nAfter frontend 4D stack:")
    diff = (hf_hidden_4d - fs2_hidden_4d).abs()
    print(f"  Diff: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Run through layers manually to track divergence
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding
    from fairseq2.models.transformer import AttentionBiasCache

    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    attn_bias_cache = AttentionBiasCache()
    fs2_per_layer_inputs = getattr(state_bag, 'per_layer_inputs', None)

    print(f"\nPer-layer divergence:")
    for layer_idx in [0, 4, 9, 14, 19, 24, 29]:  # Sample layers
        # HF
        layer_type = hf_lm.config.layer_types[layer_idx]
        cos, sin = rope(hf_hidden_4d[0], position_ids, layer_type)
        hf_layer = hf_lm.layers[layer_idx]
        hf_hidden_4d = hf_layer(
            hidden_states=hf_hidden_4d,
            per_layer_input=per_layer_inputs[:, :, layer_idx, :],
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
            past_key_values=None,
            cache_position=None,
        )

        # FS2
        fs2_layer = fs2_model.decoder.layers[layer_idx]
        fs2_layer_ple = fs2_per_layer_inputs[:, :, layer_idx, :] if fs2_per_layer_inputs is not None else None
        fs2_hidden_4d = fs2_layer(
            fs2_hidden_4d, batch_layout, attn_bias_cache,
            per_layer_input=fs2_layer_ple, state_bag=state_bag
        )

        diff = (hf_hidden_4d - fs2_hidden_4d).abs()
        print(f"  Layer {layer_idx:2d}: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # After all layers, check unstacking
    print(f"\n" + "="*80)
    print("After all layers (4D)")
    print("="*80)
    diff = (hf_hidden_4d - fs2_hidden_4d).abs()
    print(f"  Diff: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Unstack
    hf_unstacked = hf_lm.unstack(hf_hidden_4d)
    fs2_unstacked = fs2_model.decoder._unstack_altup(fs2_hidden_4d)
    print(f"\nAfter unstacking (3D):")
    diff = (hf_unstacked - fs2_unstacked).abs()
    print(f"  Diff: max={diff.max():.6e}, mean={diff.mean():.6e}")

    # Final norm
    hf_normed = hf_lm.norm(hf_unstacked)
    fs2_normed = fs2_model.decoder.layer_norm(fs2_unstacked)
    print(f"\nAfter final normalization:")
    diff = (hf_normed - fs2_normed).abs()
    print(f"  Diff: max={diff.max():.6e}, mean={diff.mean():.6e}")
