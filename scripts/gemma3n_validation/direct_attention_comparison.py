#!/usr/bin/env python3
"""
Direct comparison: Capture inputs/outputs of HF and FS2 attention modules.
No manual computation, just hook the actual forward() methods.
"""

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
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)
text = "The quick brown"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("DIRECT ATTENTION MODULE COMPARISON")
print("="*80)

text_config = hf_model.model.language_model.config

with torch.no_grad():
    # Full setup
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

    print(f"\n[COMPARING ATTENTION INPUTS]")
    diff = (hf_active_normed - fs2_active_normed).abs()
    print(f"Input diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")
    print(f"HF input[0, 3, :5]: {hf_active_normed[0, 3, :5]}")
    print(f"FS2 input[0, 3, :5]: {fs2_active_normed[0, 3, :5]}")

    # Run HF attention
    print(f"\n[HF ATTENTION]")
    hf_attn_out, _ = hf_layer0.self_attn(
        hidden_states=hf_active_normed,
        position_embeddings=position_embeddings[hf_layer0.attention_type],
    )
    print(f"Output shape: {hf_attn_out.shape}")
    print(f"Output[0, 3, :5]: {hf_attn_out[0, 3, :5]}")

    # Run FS2 attention
    print(f"\n[FS2 ATTENTION]")
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
    print(f"Output shape: {fs2_attn_out.shape}")
    print(f"Output[0, 3, :5]: {fs2_attn_out[0, 3, :5]}")

    # Compare
    print(f"\n[COMPARING ATTENTION OUTPUTS]")
    diff = (hf_attn_out - fs2_attn_out).abs()
    print(f"Output diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    if diff.max().item() < 1e-5:
        print(f"\n✅ ATTENTION OUTPUTS MATCH!")
    else:
        print(f"\n❌ ATTENTION OUTPUTS DIVERGE")
        print(f"\nThis is the root divergence we need to fix.")

    # Check for attention logit softcapping
    print(f"\n{'='*80}")
    print("CHECKING FOR SPECIAL ATTENTION FEATURES")
    print(f"{'='*80}")

    cfg = hf_layer0.self_attn.config
    attrs_to_check = [
        'attn_logit_softcapping',
        'query_pre_attn_scalar',
        'sliding_window',
        'attention_dropout',
    ]

    print(f"\n[HF Layer 0 config]")
    for attr in attrs_to_check:
        if hasattr(cfg, attr):
            val = getattr(cfg, attr)
            print(f"  {attr}: {val}")

            if attr == 'attn_logit_softcapping' and val is not None:
                print(f"\n⚠️  ATTENTION LOGIT SOFTCAPPING IS ENABLED!")
                print(f"   HF clamps attention logits: tanh(logits / {val}) * {val}")
                print(f"   This is likely the root cause - FS2 doesn't implement this!")

    print(f"\n[FS2 Layer 0 check]")
    if hasattr(fs2_layer0.self_attn, 'attn_logit_softcapping'):
        print(f"  ✓ FS2 has attn_logit_softcapping")
    else:
        print(f"  ❌ FS2 does NOT have attn_logit_softcapping attribute")

print("="*80)
