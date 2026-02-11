#!/usr/bin/env python3
"""Capture attention output before and after output projection."""

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
text = "The quick"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("CAPTURE BEFORE/AFTER OUTPUT PROJECTION")
print("="*80)

text_config = hf_model.model.language_model.config

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
    fs2_layer0 = fs2_model.decoder.layers[0]

    hf_predictions = hf_layer0.altup.predict(hf_hidden_4d)
    hf_active = hf_predictions[hf_layer0.config.altup_active_idx]
    hf_active_normed = hf_layer0.input_layernorm(hf_active)

    fs2_predictions = fs2_layer0.altup(hf_hidden_4d)
    fs2_active = fs2_predictions[fs2_layer0.altup_active_idx]
    fs2_active_normed = fs2_layer0.input_layernorm(fs2_active)

    # Check dropout settings
    print(f"\n[DROPOUT SETTINGS]")
    print(f"HF attention dropout: {hf_layer0.self_attn.attention_dropout}")
    print(f"HF training mode: {hf_layer0.self_attn.training}")
    print(f"FS2 SDPA dropout_p: {fs2_layer0.self_attn.sdpa.dropout_p}")
    print(f"FS2 training mode: {fs2_layer0.self_attn.training}")

    # Hook HF to capture before output projection
    hf_captured = {}

    import transformers.models.gemma3n.modeling_gemma3n as gemma3n_module
    original_eager = gemma3n_module.eager_attention_forward

    def eager_hook(module, query, key, value, attention_mask, dropout=0.0, scaling=None, softcap=None, **kwargs):
        attn_output, attn_weights = original_eager(module, query, key, value, attention_mask, dropout, scaling, softcap, **kwargs)
        hf_captured['before_output_proj'] = attn_output.clone()
        return attn_output, attn_weights

    gemma3n_module.eager_attention_forward = eager_hook

    hf_attn_out, _ = hf_layer0.self_attn(
        hidden_states=hf_active_normed,
        position_embeddings=position_embeddings[hf_layer0.attention_type],
    )

    gemma3n_module.eager_attention_forward = original_eager

    # Hook FS2 SDPA to capture output
    fs2_captured = {}

    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    attn_bias_cache = AttentionBiasCache()

    original_sdpa = fs2_layer0.self_attn.sdpa.forward
    def sdpa_hook(q, q_layout, k, k_layout, v, bias_cache, needs_weights=False):
        attns, weights = original_sdpa(q, q_layout, k, k_layout, v, bias_cache, needs_weights=needs_weights)
        fs2_captured['after_sdpa'] = attns.clone()  # [batch, seq, heads, dim]
        return attns, weights

    fs2_layer0.self_attn.sdpa.forward = sdpa_hook

    fs2_attn_out = fs2_layer0.self_attn(
        fs2_active_normed,
        batch_layout,
        keys=fs2_active_normed,
        keys_layout=batch_layout,
        values=fs2_active_normed,
        bias_cache=attn_bias_cache,
        state_bag=state_bag,
    )

    fs2_layer0.self_attn.sdpa.forward = original_sdpa

    print(f"\n[HF CAPTURED]")
    print(f"Before output proj shape: {hf_captured['before_output_proj'].shape}")  # [batch, heads, seq, dim]
    print(f"After full attention: {hf_attn_out.shape}")

    print(f"\n[FS2 CAPTURED]")
    print(f"After SDPA shape: {fs2_captured['after_sdpa'].shape}")  # [batch, seq, heads, dim]
    print(f"After full attention: {fs2_attn_out.shape}")

    # Compare SDPA outputs (before output projection)
    # HF: [batch, heads, seq, dim] -> transpose to [batch, seq, heads, dim]
    hf_sdpa_out = hf_captured['before_output_proj'].transpose(1, 2)
    fs2_sdpa_out = fs2_captured['after_sdpa']

    print(f"\n[COMPARING SDPA OUTPUTS]")
    diff = (hf_sdpa_out - fs2_sdpa_out).abs()
    print(f"SDPA output diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    # Compare final attention outputs
    print(f"\n[COMPARING FINAL ATTENTION OUTPUTS]")
    diff = (hf_attn_out - fs2_attn_out).abs()
    print(f"Final output diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    # Check output projection weights
    print(f"\n[OUTPUT PROJECTION WEIGHTS]")
    hf_o_weight = hf_layer0.self_attn.o_proj.weight
    fs2_o_weight = fs2_layer0.self_attn.output_proj.weight
    diff = (hf_o_weight - fs2_o_weight).abs()
    print(f"Weight diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

print("="*80)
