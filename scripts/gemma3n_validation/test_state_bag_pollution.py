#!/usr/bin/env python3
"""Test if state_bag reuse causes layer 0 divergence."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache

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
    # Setup
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

    # FS2 setup
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    seqs, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
    fs2_hidden_4d = fs2_model.decoder._stack_altup(seqs)
    fs2_per_layer_inputs = getattr(state_bag, 'per_layer_inputs', None)

    print("="*80)
    print("Test 1: Call layer with FRESH state_bag")
    print("="*80)

    # Fresh state_bag for layer
    fresh_state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    fresh_state_bag.per_layer_inputs = fs2_per_layer_inputs.clone()

    attn_bias_cache = AttentionBiasCache()
    fs2_layer = fs2_model.decoder.layers[0]
    fs2_output_fresh = fs2_layer(
        fs2_hidden_4d, batch_layout, attn_bias_cache,
        per_layer_input=fs2_per_layer_inputs[:, :, 0, :],
        state_bag=fresh_state_bag
    )

    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nRotaryEmbedding
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    rope = Gemma3nRotaryEmbedding(config=hf_lm.config)
    cos, sin = rope(hf_hidden_4d[0], position_ids, "sliding_attention")

    hf_layer = hf_lm.layers[0]
    hf_output = hf_layer(
        hidden_states=hf_hidden_4d,
        per_layer_input=per_layer_inputs[:, :, 0, :],
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
        past_key_values=None,
        cache_position=None,
    )

    diff = (hf_output - fs2_output_fresh).abs()
    print(f"With FRESH state_bag: max={diff.max():.6e}, mean={diff.mean():.6e}")

    print("\n" + "="*80)
    print("Test 2: Call layer with REUSED state_bag (from frontend)")
    print("="*80)

    attn_bias_cache2 = AttentionBiasCache()
    fs2_output_reused = fs2_layer(
        fs2_hidden_4d, batch_layout, attn_bias_cache2,
        per_layer_input=fs2_per_layer_inputs[:, :, 0, :],
        state_bag=state_bag  # Reuse original state_bag
    )

    diff = (hf_output - fs2_output_reused).abs()
    print(f"With REUSED state_bag: max={diff.max():.6e}, mean={diff.mean():.6e}")

    print("\n" + "="*80)
    print("Test 3: Check if state_bag has RoPE cache pollution")
    print("="*80)
    print(f"Fresh state_bag keys: {list(vars(fresh_state_bag).keys())}")
    print(f"Reused state_bag keys: {list(vars(state_bag).keys())}")
