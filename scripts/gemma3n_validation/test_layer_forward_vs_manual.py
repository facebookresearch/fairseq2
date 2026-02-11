#!/usr/bin/env python3
"""Compare layer.forward() vs manual component calls."""

import torch
import math
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    seqs, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
    fs2_hidden_4d = fs2_model.decoder._stack_altup(seqs)
    fs2_per_layer_inputs = getattr(state_bag, 'per_layer_inputs', None)
    ple_layer0 = fs2_per_layer_inputs[:, :, 0, :]

    fs2_layer = fs2_model.decoder.layers[0]

    print("="*80)
    print("Test 1: Call layer.forward() directly")
    print("="*80)

    attn_bias_cache = AttentionBiasCache()
    output_via_forward = fs2_layer(
        fs2_hidden_4d, batch_layout, attn_bias_cache,
        per_layer_input=ple_layer0,
        state_bag=state_bag
    )
    print(f"Output shape: {output_via_forward.shape}")
    print(f"Output: mean={output_via_forward.mean():.6f}, std={output_via_forward.std():.6f}")

    print("\n" + "="*80)
    print("Test 2: Manually reconstruct layer forward (FRESH state_bag)")
    print("="*80)

    # Use FRESH state_bag to avoid contamination from Test 1
    state_bag2 = IncrementalStateBag(max_num_steps=input_ids.size(1))
    state_bag2.per_layer_inputs = fs2_per_layer_inputs.clone()

    # Manually follow the layer forward logic
    predictions = fs2_layer.altup(fs2_hidden_4d)
    active = predictions[0]
    active_normed = fs2_layer.input_layernorm(active)
    laurel_output = fs2_layer.laurel(active_normed)

    attn_bias_cache2 = AttentionBiasCache()
    attn = fs2_layer.self_attn(
        active_normed, batch_layout, active_normed, batch_layout, active_normed,
        attn_bias_cache2, state_bag=state_bag2  # Use fresh state_bag
    )
    attn = fs2_layer.post_attention_layernorm(attn)

    attn_gated = active + attn
    attn_laurel = (attn_gated + laurel_output) / math.sqrt(2.0)

    attn_norm = fs2_layer.pre_feedforward_layernorm(attn_laurel)
    attn_ffw = fs2_layer.ffn(attn_norm)
    attn_ffw_norm = fs2_layer.post_feedforward_layernorm(attn_ffw)
    attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm

    corrected = fs2_layer.altup.correct(predictions, attn_ffw_laurel_gated)

    first = corrected[0].clone()
    if fs2_layer.altup_correct_scale:
        first = fs2_layer.altup.scale_corrected_output(first)

    first = fs2_layer.per_layer_input_gate(first)
    first = fs2_layer.hidden_activation(first)
    first = first * ple_layer0
    first = fs2_layer.per_layer_projection(first)
    first = fs2_layer.post_per_layer_input_norm(first)

    corrected[1:] += first
    output_manual = corrected

    print(f"Output shape: {output_manual.shape}")
    print(f"Output: mean={output_manual.mean():.6f}, std={output_manual.std():.6f}")

    print("\n" + "="*80)
    print("Comparison")
    print("="*80)

    diff = (output_via_forward - output_manual).abs()
    print(f"Max diff: {diff.max():.6e}")
    print(f"Mean diff: {diff.mean():.6e}")

    if diff.max() > 1e-6:
        print("\n⚠️  layer.forward() differs from manual reconstruction!")
        print("There's a bug in the layer implementation.")
    else:
        print("\n✓ layer.forward() matches manual reconstruction")
