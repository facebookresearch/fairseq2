#!/usr/bin/env python3
"""Compare PLE (Per-Layer Embeddings) computation between HF and FS2."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
text = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("PLE (PER-LAYER EMBEDDINGS) COMPARISON")
print("="*80)

with torch.no_grad():
    # HF PLE computation
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)

    print(f"\n[HF PLE COMPUTATION]")
    hf_per_layer_inputs = hf_lm.get_per_layer_inputs(input_ids)
    print(f"  Discrete PLE shape: {hf_per_layer_inputs.shape}")
    print(f"  Discrete PLE mean: {hf_per_layer_inputs.mean().item():.6f}, std: {hf_per_layer_inputs.std().item():.6f}")

    hf_per_layer_inputs = hf_lm.project_per_layer_inputs(hf_embeds, hf_per_layer_inputs)
    print(f"  Combined PLE shape: {hf_per_layer_inputs.shape}")
    print(f"  Combined PLE mean: {hf_per_layer_inputs.mean().item():.6f}, std: {hf_per_layer_inputs.std().item():.6f}")

    # FS2 PLE computation
    print(f"\n[FS2 PLE COMPUTATION]")
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))

    fs2_embeds, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)
    fs2_per_layer_inputs = state_bag.per_layer_inputs

    print(f"  PLE shape: {fs2_per_layer_inputs.shape}")
    print(f"  PLE mean: {fs2_per_layer_inputs.mean().item():.6f}, std: {fs2_per_layer_inputs.std().item():.6f}")

    # Compare PLE for each layer
    print(f"\n[COMPARING PLE FOR EACH LAYER]")
    print(f"{'Layer':<6} {'Max Diff':<15} {'Mean Diff':<15}")
    print("-" * 40)

    for layer_idx in range(min(10, config.num_layers)):  # First 10 layers
        hf_ple = hf_per_layer_inputs[:, :, layer_idx, :]
        fs2_ple = fs2_per_layer_inputs[:, :, layer_idx, :]

        diff = (hf_ple - fs2_ple).abs()
        print(f"{layer_idx:<6} {diff.max().item():<15.6e} {diff.mean().item():<15.6e}")

    # Show sample values for layer 0
    print(f"\n[LAYER 0 PLE SAMPLE VALUES]")
    print(f"  HF  layer 0 PLE: {hf_per_layer_inputs[0, 0, 0, :5].tolist()}")
    print(f"  FS2 layer 0 PLE: {fs2_per_layer_inputs[0, 0, 0, :5].tolist()}")

print("="*80)
