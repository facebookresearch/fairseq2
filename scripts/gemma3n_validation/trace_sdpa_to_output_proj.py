#!/usr/bin/env python3
"""Trace from SDPA output through output projection to find divergence."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
text = "The quick brown"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("TRACE SDPA OUTPUT → OUTPUT PROJECTION → FINAL OUTPUT")
print("="*80)

text_config = hf_model.model.language_model.config

with torch.no_grad():
    # Setup (same as before)
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
    hidden_states_0 = hf_embeds
    target_magnitude = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
    epsilon_tensor = torch.tensor(1e-5, device=device, dtype=dtype)

    temp_hidden_states = [hidden_states_0]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hidden_states_0)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden_states.append(altup_proj)

    hf_hidden_4d = torch.stack(temp_hidden_states, dim=0)

    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    position_embeddings = {}
    for layer_type in set(text_config.layer_types):
        position_embeddings[layer_type] = hf_lm.rotary_emb(hf_hidden_4d, position_ids, layer_type)

    hf_layer0 = hf_lm.layers[0]
    fs2_layer0 = fs2_model.decoder.layers[0]

    hf_predictions = hf_layer0.altup.predict(hf_hidden_4d)
    hf_active = hf_predictions[hf_layer0.config.altup_active_idx]
    hf_active_normed = hf_layer0.input_layernorm(hf_active)

    fs2_predictions = fs2_layer0.altup(hf_hidden_4d)
    fs2_active = fs2_predictions[fs2_layer0.altup_active_idx]
    fs2_active_normed = fs2_layer0.input_layernorm(fs2_active)

    # Hook HF to capture SDPA output (using torch.nn.functional.scaled_dot_product_attention)
    hf_captured = {}

    import torch.nn.functional as F
    original_sdpa = F.scaled_dot_product_attention

    def sdpa_hook(*args, **kwargs):
        output = original_sdpa(*args, **kwargs)
        hf_captured['sdpa_output'] = output.clone()
        return output

    F.scaled_dot_product_attention = sdpa_hook

    hf_attn_out, _ = hf_layer0.self_attn(
        hidden_states=hf_active_normed,
        position_embeddings=position_embeddings[hf_layer0.attention_type],
    )

    F.scaled_dot_product_attention = original_sdpa

    # Hook FS2 SDPA to capture output
    fs2_captured = {}

    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    attn_bias_cache = AttentionBiasCache()

    original_sdpa_forward = fs2_layer0.self_attn.sdpa.forward
    def fs2_sdpa_hook(q, q_layout, k, k_layout, v, bias_cache, needs_weights=False):
        attns, weights = original_sdpa_forward(q, q_layout, k, k_layout, v, bias_cache, needs_weights=needs_weights)
        fs2_captured['sdpa_output'] = attns.clone()  # [batch, seq, heads, dim]
        return attns, weights

    fs2_layer0.self_attn.sdpa.forward = fs2_sdpa_hook

    fs2_attn_out = fs2_layer0.self_attn(
        fs2_active_normed,
        batch_layout,
        keys=fs2_active_normed,
        keys_layout=batch_layout,
        values=fs2_active_normed,
        bias_cache=attn_bias_cache,
        state_bag=state_bag,
    )

    fs2_layer0.self_attn.sdpa.forward = original_sdpa_forward

    print(f"\n[CAPTURED SHAPES]")
    print(f"HF SDPA output: {hf_captured['sdpa_output'].shape}")  # likely [batch, heads, seq, dim]
    print(f"HF final attention: {hf_attn_out.shape}")
    print(f"FS2 SDPA output: {fs2_captured['sdpa_output'].shape}")  # [batch, seq, heads, dim]
    print(f"FS2 final attention: {fs2_attn_out.shape}")

    # Compare SDPA outputs (need to match layouts)
    print(f"\n[STEP 1: SDPA OUTPUT COMPARISON]")
    # HF SDPA output needs transposing to match FS2 layout
    if hf_captured['sdpa_output'].shape[1] != hf_captured['sdpa_output'].shape[2]:
        # If dim[1] != dim[2], assume HF is [batch, heads, seq, dim]
        hf_sdpa_transposed = hf_captured['sdpa_output'].transpose(1, 2)
    else:
        hf_sdpa_transposed = hf_captured['sdpa_output']

    diff = (hf_sdpa_transposed - fs2_captured['sdpa_output']).abs()
    print(f"SDPA output diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    # Check for NaN/Inf
    hf_has_nan = torch.isnan(hf_sdpa_transposed).any()
    fs2_has_nan = torch.isnan(fs2_captured['sdpa_output']).any()
    hf_has_inf = torch.isinf(hf_sdpa_transposed).any()
    fs2_has_inf = torch.isinf(fs2_captured['sdpa_output']).any()
    print(f"NaN/Inf check: HF (nan={hf_has_nan}, inf={hf_has_inf}), FS2 (nan={fs2_has_nan}, inf={fs2_has_inf})")

    # Find where max diff occurs
    max_diff_idx = diff.argmax()
    max_diff_pos = torch.unravel_index(max_diff_idx, diff.shape)
    print(f"\nMax diff at position [batch={max_diff_pos[0]}, seq={max_diff_pos[1]}, head={max_diff_pos[2]}, dim={max_diff_pos[3]}]")
    print(f"  HF value: {hf_sdpa_transposed[max_diff_pos].item():.6f}")
    print(f"  FS2 value: {fs2_captured['sdpa_output'][max_diff_pos].item():.6f}")

    # Show per-position statistics
    print(f"\nDiff distribution:")
    print(f"  Positions with diff > 1e-5: {(diff > 1e-5).sum().item()} / {diff.numel()}")
    print(f"  Positions with diff > 1e-3: {(diff > 1e-3).sum().item()} / {diff.numel()}")
    print(f"  Positions with diff > 0.1: {(diff > 0.1).sum().item()} / {diff.numel()}")

    if diff.max().item() < 1e-5:
        print("✅ SDPA outputs match in real forward pass")
    else:
        print("❌ SDPA outputs DIVERGE in real forward pass")
        print(f"\nSample values at [0,0,0,:5]:")
        print(f"  HF SDPA: {hf_sdpa_transposed[0,0,0,:5]}")
        print(f"  FS2 SDPA: {fs2_captured['sdpa_output'][0,0,0,:5]}")
        print(f"\nSample values at max diff position's head [0,{max_diff_pos[1]},{max_diff_pos[2]},:5]:")
        print(f"  HF SDPA: {hf_sdpa_transposed[0,max_diff_pos[1],max_diff_pos[2],:5]}")
        print(f"  FS2 SDPA: {fs2_captured['sdpa_output'][0,max_diff_pos[1],max_diff_pos[2],:5]}")

    # Compare final attention outputs
    print(f"\n[STEP 2: FINAL ATTENTION OUTPUT COMPARISON]")
    diff = (hf_attn_out - fs2_attn_out).abs()
    print(f"Final output diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    if diff.max().item() < 1e-5:
        print("✅ Final attention outputs match")
    else:
        print("❌ Final attention outputs DIVERGE")
        print(f"\nSample values:")
        print(f"  HF final[0,0,:5]: {hf_attn_out[0,0,:5]}")
        print(f"  FS2 final[0,0,:5]: {fs2_attn_out[0,0,:5]}")

    # Check output projection weights
    print(f"\n[STEP 3: OUTPUT PROJECTION WEIGHTS]")
    hf_o_weight = hf_layer0.self_attn.o_proj.weight
    fs2_o_weight = fs2_layer0.self_attn.output_proj.weight
    diff = (hf_o_weight - fs2_o_weight).abs()
    print(f"Weight shape: HF={hf_o_weight.shape}, FS2={fs2_o_weight.shape}")
    print(f"Weight diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    # Manually apply output projection to verify
    print(f"\n[STEP 4: MANUAL OUTPUT PROJECTION]")
    # HF applies o_proj to [batch, heads, seq, dim] after transpose
    # Need to flatten heads and dim for linear projection
    hf_sdpa_for_proj = hf_captured['sdpa_output'].transpose(1, 2).contiguous()  # [batch, seq, heads, dim]
    batch, seq, heads, dim = hf_sdpa_for_proj.shape
    hf_sdpa_flat = hf_sdpa_for_proj.reshape(batch, seq, heads * dim)
    hf_manual_proj = F.linear(hf_sdpa_flat, hf_o_weight)

    fs2_sdpa_flat = fs2_captured['sdpa_output'].reshape(batch, seq, heads * dim)
    fs2_manual_proj = F.linear(fs2_sdpa_flat, fs2_o_weight)

    print(f"Manual projection shapes: HF={hf_manual_proj.shape}, FS2={fs2_manual_proj.shape}")
    diff = (hf_manual_proj - fs2_manual_proj).abs()
    print(f"Manual projection diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    # Check if manual projection matches actual outputs
    print(f"\nVerify manual projection matches actual:")
    diff_hf = (hf_manual_proj - hf_attn_out).abs()
    diff_fs2 = (fs2_manual_proj - fs2_attn_out).abs()
    print(f"  HF manual vs actual: max={diff_hf.max().item():.6e}")
    print(f"  FS2 manual vs actual: max={diff_fs2.max().item():.6e}")

print("="*80)
