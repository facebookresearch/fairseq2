"""Test 3: Isolate which Layer 0 component diverges."""

import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache

device = torch.device("cpu")
dtype = torch.float32
model_name = "google/gemma-3n-E2B-it"

print("Loading models...")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
text = "The quick brown"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
print(f"Input: {text!r}\n")

seq_lens = [input_ids.shape[1]]
batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
state_bag = IncrementalStateBag(input_ids.shape[1])

print("=" * 80)
print("TEST 3: Isolate Layer 0 Component Divergence")
print("=" * 80)

with torch.no_grad():
    # Get embeddings
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
    fs2_embeds, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)

    # Create 4D inputs
    target_magnitude = torch.mean(hf_embeds**2, dim=-1, keepdim=True) ** 0.5
    epsilon = torch.tensor(1e-5, device=device, dtype=dtype)

    temp_hidden_states = [hf_embeds]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hf_embeds)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden_states.append(altup_proj)

    hf_hidden_4d = torch.stack(temp_hidden_states, dim=0)
    fs2_hidden_4d = fs2_model.decoder._stack_altup(fs2_embeds)

    # Setup
    hf_layer0 = hf_lm.layers[0]
    fs2_layer0 = fs2_model.decoder.layers[0]
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    text_config = hf_model.model.language_model.config
    position_embeddings = {}
    for layer_type in set(text_config.layer_types):
        position_embeddings[layer_type] = hf_lm.rotary_emb(hf_hidden_4d, position_ids, layer_type)

    attn_bias_cache = AttentionBiasCache()
    layer_ple = state_bag.per_layer_inputs[:, :, 0, :]

    def compare(name, hf_val, fs2_val):
        """Compare tensors and report."""
        # Handle 4D vs 3D comparison
        if hf_val.ndim == 4 and fs2_val.ndim == 3:
            hf_val = hf_val[0]  # Extract active dimension
        elif hf_val.ndim == 3 and fs2_val.ndim == 4:
            fs2_val = fs2_val[0]

        diff = (hf_val - fs2_val).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        status = "✅" if max_diff < 1e-5 else "❌"
        print(f"{status} {name:30s} max={max_diff:.6e}, mean={mean_diff:.6e}")
        return max_diff < 1e-5

    # ========================================================================
    # Step-by-step comparison
    # ========================================================================

    print("\n1. AltUp Predict")
    print("-" * 80)
    hf_predictions = hf_layer0.altup.predict(hf_hidden_4d)
    fs2_predictions = fs2_layer0.altup(fs2_hidden_4d)
    compare("AltUp predictions (4D)", hf_predictions, fs2_predictions)

    hf_active = hf_predictions[0]
    fs2_active = fs2_predictions[0]
    compare("Active prediction", hf_active, fs2_active)

    print("\n2. Input Normalization")
    print("-" * 80)
    hf_normed = hf_layer0.input_layernorm(hf_active)
    fs2_normed = fs2_layer0.input_layernorm(fs2_active)
    compare("Input norm", hf_normed, fs2_normed)

    print("\n3. LAuReL")
    print("-" * 80)
    hf_laurel = hf_layer0.laurel(hf_normed)
    fs2_laurel = fs2_layer0.laurel(fs2_normed)
    compare("LAuReL output", hf_laurel, fs2_laurel)

    print("\n4. Attention")
    print("-" * 80)
    hf_attn, _ = hf_layer0.self_attn(
        hidden_states=hf_normed,
        position_embeddings=position_embeddings[hf_layer0.attention_type],
    )
    fs2_attn = fs2_layer0.self_attn(
        fs2_normed,
        batch_layout,
        keys=fs2_normed,
        keys_layout=batch_layout,
        values=fs2_normed,
        bias_cache=attn_bias_cache,
        state_bag=state_bag,
    )
    compare("Attention output", hf_attn, fs2_attn)

    print("\n5. Post-Attention Norm")
    print("-" * 80)
    hf_attn_norm = hf_layer0.post_attention_layernorm(hf_attn)
    fs2_attn_norm = fs2_layer0.post_attention_layernorm(fs2_attn)
    compare("Post-attn norm", hf_attn_norm, fs2_attn_norm)

    print("\n6. Gating & LAuReL Combine")
    print("-" * 80)
    hf_attn_gated = hf_active + hf_attn_norm
    fs2_attn_gated = fs2_active + fs2_attn_norm
    compare("Attn gated", hf_attn_gated, fs2_attn_gated)

    hf_attn_laurel = (hf_attn_gated + hf_laurel) / math.sqrt(2)
    fs2_attn_laurel = (fs2_attn_gated + fs2_laurel) / math.sqrt(2.0)
    compare("Attn + LAuReL", hf_attn_laurel, fs2_attn_laurel)

    print("\n7. Pre-FFN Norm")
    print("-" * 80)
    hf_pre_ffn = hf_layer0.pre_feedforward_layernorm(hf_attn_laurel)
    fs2_pre_ffn = fs2_layer0.pre_feedforward_layernorm(fs2_attn_laurel)
    compare("Pre-FFN norm", hf_pre_ffn, fs2_pre_ffn)

    print("\n8. FFN (AltUp MLP)")
    print("-" * 80)
    hf_ffn = hf_layer0.mlp(hf_pre_ffn)
    fs2_ffn = fs2_layer0.ffn(fs2_pre_ffn)
    compare("FFN output", hf_ffn, fs2_ffn)

    print("\n9. Post-FFN Norm")
    print("-" * 80)
    hf_ffn_norm = hf_layer0.post_feedforward_layernorm(hf_ffn)
    fs2_ffn_norm = fs2_layer0.post_feedforward_layernorm(fs2_ffn)
    compare("Post-FFN norm", hf_ffn_norm, fs2_ffn_norm)

    print("\n10. Combine FFN")
    print("-" * 80)
    hf_combined = hf_attn_laurel + hf_ffn_norm
    fs2_combined = fs2_attn_laurel + fs2_ffn_norm
    compare("Attn+LAuReL+FFN", hf_combined, fs2_combined)

    print("\n11. AltUp Correct")
    print("-" * 80)
    hf_corrected = hf_layer0.altup.correct(hf_predictions, hf_combined)
    fs2_corrected = fs2_layer0.altup.correct(fs2_predictions, fs2_combined)
    compare("Corrected predictions (4D)", hf_corrected, fs2_corrected)

    print("\n12. Extract & Scale")
    print("-" * 80)
    hf_first = hf_corrected[0].clone()
    fs2_first = fs2_corrected[0].clone()

    if hf_layer0.config.altup_correct_scale:
        hf_first = hf_layer0.altup.scale_corrected_output(hf_first)
    if fs2_layer0.altup_correct_scale:
        fs2_first = fs2_layer0.altup.scale_corrected_output(fs2_first)

    compare("First prediction (scaled)", hf_first, fs2_first)

    print("\n13. PLE Gating")
    print("-" * 80)
    hf_ple_gate = hf_layer0.per_layer_input_gate(hf_first)
    hf_ple_gate = hf_layer0.act_fn(hf_ple_gate)
    hf_ple_gate = torch.multiply(hf_ple_gate, layer_ple)

    fs2_ple_gate = fs2_layer0.per_layer_input_gate(fs2_first)
    fs2_ple_gate = fs2_layer0.hidden_activation(fs2_ple_gate)
    fs2_ple_gate = fs2_ple_gate * layer_ple

    compare("PLE gated", hf_ple_gate, fs2_ple_gate)

    print("\n14. PLE Projection & Norm")
    print("-" * 80)
    hf_ple_proj = hf_layer0.per_layer_projection(hf_ple_gate)
    fs2_ple_proj = fs2_layer0.per_layer_projection(fs2_ple_gate)
    compare("PLE projection", hf_ple_proj, fs2_ple_proj)

    hf_ple_norm = hf_layer0.post_per_layer_input_norm(hf_ple_proj)
    fs2_ple_norm = fs2_layer0.post_per_layer_input_norm(fs2_ple_proj)
    compare("PLE norm", hf_ple_norm, fs2_ple_norm)

    print("\n15. Final Update")
    print("-" * 80)
    hf_corrected[1:] += hf_ple_norm
    fs2_corrected[1:] += fs2_ple_norm
    compare("Final output (4D)", hf_corrected, fs2_corrected)

print("\n" + "=" * 80)
print("First diverging component indicates root cause")
print("=" * 80)
