"""Test 12: Isolate layer 4 (first global layer) divergence."""

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
print("TEST 12: Isolate Layer 4 (First Global Layer) Divergence")
print("=" * 80)

with torch.no_grad():
    # Get embeddings
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)
    fs2_embeds, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)

    # Stack to 4D
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
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    text_config = hf_model.model.language_model.config
    position_embeddings = {}
    for layer_type in set(text_config.layer_types):
        position_embeddings[layer_type] = hf_lm.rotary_emb(hf_hidden_4d, position_ids, layer_type)

    attn_bias_cache = AttentionBiasCache()
    per_layer_inputs = state_bag.per_layer_inputs

    # Run through layers 0-3 to get to layer 4 input
    print("Running layers 0-3 (local layers)...")
    hf_hidden = hf_hidden_4d
    fs2_hidden = fs2_hidden_4d

    for layer_idx in range(4):
        hf_layer = hf_lm.layers[layer_idx]
        fs2_layer = fs2_model.decoder.layers[layer_idx]
        layer_ple = per_layer_inputs[:, :, layer_idx, :]

        hf_hidden = hf_layer(
            hf_hidden,
            position_embeddings=position_embeddings[hf_layer.attention_type],
            per_layer_input=layer_ple,
        )
        fs2_hidden = fs2_layer(
            fs2_hidden,
            batch_layout,
            attn_bias_cache,
            per_layer_input=layer_ple,
            state_bag=state_bag,
        )

    diff = (hf_hidden[0] - fs2_hidden[0]).abs()
    print(f"After layer 3: max diff = {diff.max().item():.6e}")
    print()

    # Now test layer 4 step-by-step
    print("=" * 80)
    print("Layer 4 (Global Layer) Step-by-Step")
    print("=" * 80)

    hf_layer4 = hf_lm.layers[4]
    fs2_layer4 = fs2_model.decoder.layers[4]
    layer4_ple = per_layer_inputs[:, :, 4, :]

    print(f"Layer type: {hf_layer4.attention_type}")
    print(f"HF FFN type: {type(hf_layer4.mlp).__name__}")
    print(f"FS2 FFN type: {type(fs2_layer4.ffn).__name__}")
    print()

    def compare(name, hf_val, fs2_val):
        if hf_val.ndim == 4 and fs2_val.ndim == 3:
            hf_val = hf_val[0]
        elif hf_val.ndim == 3 and fs2_val.ndim == 4:
            fs2_val = fs2_val[0]
        diff = (hf_val - fs2_val).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        status = "✅" if max_diff < 1e-4 else "❌"
        print(f"{status} {name:30s} max={max_diff:.6e}, mean={mean_diff:.6e}")
        return max_diff < 1e-4

    # Step-by-step
    print("\n1. AltUp Predict")
    print("-" * 80)
    hf_pred = hf_layer4.altup.predict(hf_hidden)
    fs2_pred = fs2_layer4.altup(fs2_hidden)
    compare("AltUp predictions", hf_pred, fs2_pred)

    hf_active = hf_pred[0]
    fs2_active = fs2_pred[0]
    compare("Active prediction", hf_active, fs2_active)

    print("\n2. Input Norm")
    print("-" * 80)
    hf_normed = hf_layer4.input_layernorm(hf_active)
    fs2_normed = fs2_layer4.input_layernorm(fs2_active)
    compare("Input norm", hf_normed, fs2_normed)

    print("\n3. LAuReL")
    print("-" * 80)
    hf_laurel = hf_layer4.laurel(hf_normed)
    fs2_laurel = fs2_layer4.laurel(fs2_normed)
    compare("LAuReL", hf_laurel, fs2_laurel)

    print("\n4. Attention")
    print("-" * 80)
    hf_attn, _ = hf_layer4.self_attn(
        hidden_states=hf_normed,
        position_embeddings=position_embeddings[hf_layer4.attention_type],
    )
    fs2_attn = fs2_layer4.self_attn(
        fs2_normed,
        batch_layout,
        keys=fs2_normed,
        keys_layout=batch_layout,
        values=fs2_normed,
        bias_cache=attn_bias_cache,
        state_bag=state_bag,
    )
    compare("Attention", hf_attn, fs2_attn)

    print("\n5. Post-Attention Norm")
    print("-" * 80)
    hf_attn_norm = hf_layer4.post_attention_layernorm(hf_attn)
    fs2_attn_norm = fs2_layer4.post_attention_layernorm(fs2_attn)
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
    hf_pre_ffn = hf_layer4.pre_feedforward_layernorm(hf_attn_laurel)
    fs2_pre_ffn = fs2_layer4.pre_feedforward_layernorm(fs2_attn_laurel)
    compare("Pre-FFN norm", hf_pre_ffn, fs2_pre_ffn)

    print("\n8. FFN (GLU for global layer)")
    print("-" * 80)
    hf_ffn = hf_layer4.mlp(hf_pre_ffn)
    fs2_ffn = fs2_layer4.ffn(fs2_pre_ffn)
    compare("FFN output", hf_ffn, fs2_ffn)

    print("\n9. Post-FFN Norm")
    print("-" * 80)
    hf_ffn_norm = hf_layer4.post_feedforward_layernorm(hf_ffn)
    fs2_ffn_norm = fs2_layer4.post_feedforward_layernorm(fs2_ffn)
    compare("Post-FFN norm", hf_ffn_norm, fs2_ffn_norm)

print("\n" + "=" * 80)
print("First diverging component in layer 4 indicates root cause")
print("=" * 80)
