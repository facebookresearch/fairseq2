#!/usr/bin/env python3
"""Debug why FS2 TorchSDPA isn't calling scaled_dot_product_attention."""

import torch
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache
from fairseq2.models.transformer.sdpa.torch import TorchSDPA
from transformers import AutoModelForCausalLM, AutoTokenizer

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
print("DEBUG FS2 TorchSDPA EXECUTION")
print("="*80)

# Switch to TorchSDPA
for layer in fs2_model.decoder.layers:
    old_sdpa = layer.self_attn.sdpa
    layer.self_attn.sdpa = TorchSDPA(old_sdpa.bias, dropout_p=old_sdpa.dropout_p)

fs2_layer0 = fs2_model.decoder.layers[0]

# Hook TorchSDPA.forward to see what's happening
original_torch_sdpa_forward = TorchSDPA.forward

def torch_sdpa_forward_hook(self, q, q_layout, k, k_layout, v, bias_cache, *, needs_weights=False):
    print(f"\n[TorchSDPA.forward CALLED]")
    print(f"  Q shape: {q.shape}")
    print(f"  K shape: {k.shape}")
    print(f"  V shape: {v.shape}")
    print(f"  q_layout.packed: {q_layout.packed}")
    print(f"  q_layout.padded: {q_layout.padded}")
    print(f"  k_layout.packed: {k_layout.packed}")
    print(f"  k_layout.padded: {k_layout.padded}")

    # Check is_causal condition
    from fairseq2.models.transformer.attention_bias import CausalAttentionBias
    is_causal = False

    if isinstance(self.bias, CausalAttentionBias):
        print(f"  bias is CausalAttentionBias")
        print(f"  bias.attn_window_len: {self.bias.attn_window_len}")

        if self.bias.attn_window_len is None:
            print(f"  attn_window_len is None - checking for is_causal")
            full_q = not q_layout.packed and not q_layout.padded
            full_k = not k_layout.packed and not k_layout.padded
            print(f"  full_q: {full_q}, full_k: {full_k}")

            if full_q and full_k:
                q_len = q.size(1)
                k_len = k.size(1)
                is_causal = q_len == k_len
                print(f"  q_len: {q_len}, k_len: {k_len}, is_causal: {is_causal}")
        else:
            print(f"  attn_window_len is {self.bias.attn_window_len} - will use explicit bias")

    print(f"  Final is_causal: {is_causal}")

    # Call original
    result = original_torch_sdpa_forward(self, q, q_layout, k, k_layout, v, bias_cache, needs_weights=needs_weights)

    print(f"  Output shape: {result[0].shape}")

    return result

TorchSDPA.forward = torch_sdpa_forward_hook

# Also hook scaled_dot_product_attention
call_count = [0]
original_torch_sdpa = torch.nn.functional.scaled_dot_product_attention

def sdpa_hook(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    call_count[0] += 1
    print(f"\n[scaled_dot_product_attention CALL #{call_count[0]}]")
    print(f"  is_causal: {is_causal}")
    print(f"  attn_mask: {attn_mask.shape if attn_mask is not None else None}")

    return original_torch_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

torch.nn.functional.scaled_dot_product_attention = sdpa_hook

with torch.no_grad():
    text_config = hf_model.model.language_model.config

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

    fs2_predictions = fs2_layer0.altup(hf_hidden_4d)
    fs2_active = fs2_predictions[fs2_layer0.altup_active_idx]
    fs2_active_normed = fs2_layer0.input_layernorm(fs2_active)

    print(f"\n{'='*80}")
    print("RUNNING FS2 ATTENTION")
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

    print(f"\n{'='*80}")
    print(f"Total scaled_dot_product_attention calls: {call_count[0]}")
    print(f"{'='*80}")

torch.nn.functional.scaled_dot_product_attention = original_torch_sdpa
TorchSDPA.forward = original_torch_sdpa_forward

print("="*80)
