"""Test 2: Layer 0 complete forward pass - find which component diverges."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache

device = torch.device("cpu")
dtype = torch.float32
model_name = "google/gemma-3n-E2B-it"

# Load models
print("Loading models...")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

# Tokenize input
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
text = "The quick brown"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
print(f"Input: {text!r}")
print(f"Shape: {input_ids.shape}\n")

# Prepare FS2 infrastructure
seq_lens = [input_ids.shape[1]]
batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
state_bag = IncrementalStateBag(input_ids.shape[1])
attn_bias_cache = AttentionBiasCache()

print("=" * 80)
print("TEST 2: Layer 0 Complete Forward Pass")
print("=" * 80)

with torch.no_grad():
    # Get frontend outputs (already verified to match)
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)

    fs2_embeds, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)

    print(f"✅ Frontend match confirmed (from Test 1)")
    print()

    # Prepare 4D inputs for AltUp processing
    print("Preparing 4D inputs (AltUp stacking)...")

    # HF Layer 0 setup
    hf_layer0 = hf_lm.layers[0]
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

    # HF: Manual stacking with magnitude normalization
    target_magnitude = torch.mean(hf_embeds**2, dim=-1, keepdim=True) ** 0.5
    epsilon_tensor = torch.tensor(1e-5, device=device, dtype=dtype)

    temp_hidden_states = [hf_embeds]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hf_embeds)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden_states.append(altup_proj)

    hf_hidden_4d = torch.stack(temp_hidden_states, dim=0)

    # FS2: Using decoder's _stack_altup method
    fs2_hidden_4d = fs2_model.decoder._stack_altup(fs2_embeds)

    # Verify 4D inputs match
    input_4d_diff = (hf_hidden_4d - fs2_hidden_4d).abs()
    print(f"4D input diff: max={input_4d_diff.max().item():.6e}, mean={input_4d_diff.mean().item():.6e}")
    if input_4d_diff.max() > 1e-5:
        print("⚠️  WARNING: 4D inputs differ - this will affect layer comparison")
    else:
        print("✅ 4D inputs match")
    print()

    # Get RoPE for all layer types
    text_config = hf_model.model.language_model.config
    position_embeddings = {}
    for layer_type in set(text_config.layer_types):
        position_embeddings[layer_type] = hf_lm.rotary_emb(hf_hidden_4d, position_ids, layer_type)

    # Store intermediate outputs
    intermediates = {}

    # ========================================================================
    # HF Layer 0 Forward (with intermediate captures)
    # ========================================================================
    print("Running HF Layer 0 forward...")

    # 0. AltUp predict
    hf_predictions = hf_layer0.altup.predict(hf_hidden_4d)
    hf_active = hf_predictions[0]  # altup_active_idx = 0
    intermediates['hf_active'] = hf_active

    # 1. Input normalization
    hf_normed = hf_layer0.input_layernorm(hf_active)
    intermediates['hf_after_input_norm'] = hf_normed

    # 2. LAuReL (takes normalized input, returns residual)
    hf_laurel_out = hf_layer0.laurel(hf_normed)
    intermediates['hf_laurel_output'] = hf_laurel_out

    # 3. Attention
    hf_attn_out, _ = hf_layer0.self_attn(
        hidden_states=hf_normed,
        position_embeddings=position_embeddings[hf_layer0.attention_type],
    )
    intermediates['hf_after_attn'] = hf_attn_out

    # 4. Post-attention normalization
    hf_attn_normed = hf_layer0.post_attention_layernorm(hf_attn_out)
    intermediates['hf_after_attn_norm'] = hf_attn_normed

    # 5. Gating: active + attn
    hf_attn_gated = hf_active + hf_attn_normed
    intermediates['hf_attn_gated'] = hf_attn_gated

    # 6. Combine with LAuReL: (attn_gated + laurel) / sqrt(2)
    import math
    hf_attn_laurel = (hf_attn_gated + hf_laurel_out) / math.sqrt(2)
    intermediates['hf_attn_laurel'] = hf_attn_laurel

    # 7. Pre-FFN normalization
    hf_pre_ffn_normed = hf_layer0.pre_feedforward_layernorm(hf_attn_laurel)
    intermediates['hf_after_pre_ffn_norm'] = hf_pre_ffn_normed

    # 8. FFN
    hf_ffn_out = hf_layer0.mlp(hf_pre_ffn_normed)
    intermediates['hf_after_ffn'] = hf_ffn_out

    # 9. Post-FFN normalization
    hf_post_ffn_normed = hf_layer0.post_feedforward_layernorm(hf_ffn_out)
    intermediates['hf_after_post_ffn_norm'] = hf_post_ffn_normed

    # 10. Combine: attn_laurel + ffn_norm
    hf_attn_ffw_laurel_gated = hf_attn_laurel + hf_post_ffn_normed
    intermediates['hf_attn_ffw_laurel_gated'] = hf_attn_ffw_laurel_gated

    # 11. AltUp correction
    hf_corrected_predictions = hf_layer0.altup.correct(hf_predictions, hf_attn_ffw_laurel_gated)
    intermediates['hf_corrected_predictions'] = hf_corrected_predictions

    # 12. Extract first prediction and optionally scale
    hf_first_pred = hf_corrected_predictions[0].clone()
    if hf_layer0.config.altup_correct_scale:
        hf_first_pred = hf_layer0.altup.scale_corrected_output(hf_first_pred)
    intermediates['hf_first_pred_scaled'] = hf_first_pred

    # 13. PLE gating with activation
    per_layer_input = state_bag.per_layer_inputs[:, :, 0, :]  # Layer 0 PLE
    hf_ple_gated = hf_layer0.per_layer_input_gate(hf_first_pred)
    hf_ple_gated = hf_layer0.act_fn(hf_ple_gated)
    hf_ple_gated = torch.multiply(hf_ple_gated, per_layer_input)
    intermediates['hf_ple_gated'] = hf_ple_gated

    # 14. PLE projection
    hf_ple_proj = hf_layer0.per_layer_projection(hf_ple_gated)
    intermediates['hf_ple_proj'] = hf_ple_proj

    # 15. Post-PLE normalization
    hf_ple_normed = hf_layer0.post_per_layer_input_norm(hf_ple_proj)
    intermediates['hf_ple_normed'] = hf_ple_normed

    # 16. Add to other predictions (corrected_predictions[1:] += first_prediction)
    hf_corrected_predictions[1:] += hf_ple_normed

    # Final output is the corrected 4D predictions
    hf_output = hf_corrected_predictions
    intermediates['hf_final_output'] = hf_output

    print("✅ HF Layer 0 complete\n")

    # ========================================================================
    # FS2 Layer 0 Forward (complete)
    # ========================================================================
    print("Running FS2 Layer 0 forward...")

    # Extract PLE for layer 0
    layer_ple = state_bag.per_layer_inputs[:, :, 0, :]

    # FS2 Layer 0 setup
    fs2_layer0 = fs2_model.decoder.layers[0]
    attn_bias_cache = AttentionBiasCache()

    # Run layer forward (4D → 4D)
    fs2_output = fs2_layer0(
        fs2_hidden_4d,
        batch_layout,
        attn_bias_cache,
        per_layer_input=layer_ple,
        state_bag=state_bag,
    )

    print("✅ FS2 Layer 0 complete\n")

    # ========================================================================
    # Compare final outputs
    # ========================================================================
    print("=" * 80)
    print("COMPARISON: HF vs FS2 Layer 0 Output (4D)")
    print("=" * 80)

    # Both outputs are 4D: [4, B, S, M]
    diff = (hf_output - fs2_output).abs()
    print(f"Max diff:  {diff.max().item():.6e}")
    print(f"Mean diff: {diff.mean().item():.6e}")
    print(f"Shape: HF={hf_output.shape}, FS2={fs2_output.shape}")
    print()

    if diff.max() < 1e-5:
        print("✅ MATCH - Layer 0 outputs are identical")
    else:
        print(f"❌ DIVERGENCE detected")
        print()
        print("Per-version max diff:")
        for version in range(4):
            version_diff = diff[version].max()
            print(f"  Version {version}: {version_diff:.6e}")
        print()
        print("Sample values at version 0, [0, 0, :5]:")
        print(f"  HF:  {hf_output[0, 0, 0, :5]}")
        print(f"  FS2: {fs2_output[0, 0, 0, :5]}")
        print(f"  Diff: {diff[0, 0, 0, :5]}")
        print()
        print("Per-position max diff (version 0):")
        for pos in range(min(4, input_ids.shape[1])):
            pos_diff = diff[0, 0, pos].max()
            print(f"  Position {pos}: {pos_diff:.6e}")
