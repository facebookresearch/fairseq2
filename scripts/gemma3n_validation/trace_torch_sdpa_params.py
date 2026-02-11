#!/usr/bin/env python3
"""Check what is_causal and bias TorchSDPA is using."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache
from fairseq2.models.transformer.sdpa.torch import TorchSDPA

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
print("TRACE TorchSDPA is_causal AND bias")
print("="*80)

text_config = hf_model.model.language_model.config

# Switch to TorchSDPA
for layer in fs2_model.decoder.layers:
    old_sdpa = layer.self_attn.sdpa
    layer.self_attn.sdpa = TorchSDPA(old_sdpa.bias, dropout_p=old_sdpa.dropout_p)

fs2_layer0 = fs2_model.decoder.layers[0]

print(f"\nLayer 0 SDPA bias type: {type(fs2_layer0.self_attn.sdpa.bias)}")
print(f"Layer 0 SDPA bias: {fs2_layer0.self_attn.sdpa.bias}")

# Check if it's CausalAttentionBias
from fairseq2.models.transformer.attention_bias import CausalAttentionBias
if isinstance(fs2_layer0.self_attn.sdpa.bias, CausalAttentionBias):
    print(f"Is CausalAttentionBias: Yes")
    print(f"attn_window_len: {fs2_layer0.self_attn.sdpa.bias.attn_window_len}")
else:
    print(f"Is CausalAttentionBias: No")

# Hook TorchSDPA to see what it passes to scaled_dot_product_attention
captured = {}

original_torch_sdpa = torch.nn.functional.scaled_dot_product_attention

def torch_sdpa_hook(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    captured['attn_mask'] = attn_mask.clone() if attn_mask is not None else None
    captured['is_causal'] = is_causal
    captured['dropout_p'] = dropout_p

    print(f"\n[CAPTURED scaled_dot_product_attention CALL]")
    print(f"  Q shape: {query.shape}")
    print(f"  K shape: {key.shape}")
    print(f"  V shape: {value.shape}")
    print(f"  is_causal: {is_causal}")
    print(f"  dropout_p: {dropout_p}")
    print(f"  attn_mask: {attn_mask.shape if attn_mask is not None else None}")

    if attn_mask is not None:
        print(f"  attn_mask content:")
        if attn_mask.dim() == 2:
            print(f"    {attn_mask}")
        elif attn_mask.dim() == 3:
            print(f"    {attn_mask[0]}")
        elif attn_mask.dim() == 4:
            print(f"    {attn_mask[0, 0]}")

    return original_torch_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

torch.nn.functional.scaled_dot_product_attention = torch_sdpa_hook

with torch.no_grad():
    # Setup
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

    hf_predictions = hf_layer0.altup.predict(hf_hidden_4d)
    hf_active = hf_predictions[hf_layer0.config.altup_active_idx]
    hf_active_normed = hf_layer0.input_layernorm(hf_active)

    fs2_predictions = fs2_layer0.altup(hf_hidden_4d)
    fs2_active = fs2_predictions[fs2_layer0.altup_active_idx]
    fs2_active_normed = fs2_layer0.input_layernorm(fs2_active)

    print(f"\n{'='*80}")
    print("HF CALL TO scaled_dot_product_attention")
    print(f"{'='*80}")

    hf_attn_out, _ = hf_layer0.self_attn(
        hidden_states=hf_active_normed,
        position_embeddings=position_embeddings[hf_layer0.attention_type],
    )

    print(f"\n{'='*80}")
    print("FS2 CALL TO scaled_dot_product_attention")
    print(f"{'='*80}")

    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    attn_bias_cache = AttentionBiasCache()

    fs2_attn_out = fs2_layer0.self_attn(
        fs2_active_normed,
        batch_layout,
        keys=fs2_active_normed,
        keys_layout=batch_layout,
        values=fs2_active_normed,
        bias_cache=attn_bias_cache,
        state_bag=state_bag,
    )

torch.nn.functional.scaled_dot_product_attention = original_torch_sdpa

print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")

if 'attn_mask' in captured:
    print(f"\nDid HF and FS2 call SDPA with same parameters?")
    print(f"  (Check the two captured calls above)")

print("="*80)
