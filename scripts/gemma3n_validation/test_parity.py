#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full parity test between fairseq2 Gemma3n and HuggingFace implementation."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout


def main() -> None:
    print("="*80)
    print("GEMMA3N PARITY TEST (With KV Sharing)")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"NOTE: Testing with use_cache=True to enable KV sharing in both HF and FS2")
    print(f"      HF KV sharing activates when use_cache=True")
    print(f"      FS2 KV sharing is always active")

    # Load HuggingFace model
    print("\n[1/5] Loading HuggingFace model...")
    hf_model_id = "google/gemma-3n-E2B-it"

    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=torch.float32,
        device_map=device,
    )
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    print(f"✓ Loaded HF model: {hf_model_id}")
    print(f"  Total parameters: {sum(p.numel() for p in hf_model.parameters()):,}")
    print(f"  Activation sparsity: Enabled (0.95 for first 10 layers)")

    # Create fairseq2 model
    print("\n[2/5] Creating fairseq2 model...")
    config = get_gemma3n_e2b_config()
    fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32)
    fs2_model.eval()
    print(f"✓ Created fairseq2 model")
    print(f"  Total parameters: {sum(p.numel() for p in fs2_model.parameters()):,}")
    print(f"  Activation sparsity: Enabled (0.95 for first 10 layers)")
    print(f"  KV sharing: Enabled (layers 15-29 share from layers 13-14)")

    # Convert checkpoint
    print("\n[3/5] Converting HuggingFace checkpoint to fairseq2...")
    hf_state_dict = hf_model.state_dict()
    print(f"  HF state dict keys: {len(hf_state_dict)}")

    fs2_state_dict = convert_gemma3n_state_dict(hf_state_dict, config)
    print(f"  Converted state dict keys: {len(fs2_state_dict)}")

    # Load into fairseq2 model
    missing, unexpected = fs2_model.load_state_dict(fs2_state_dict, strict=False)

    if missing:
        print(f"\n⚠️  Missing keys ({len(missing)}):")
        for key in sorted(missing)[:10]:
            print(f"    - {key}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    if unexpected:
        print(f"\n⚠️  Unexpected keys ({len(unexpected)}):")
        for key in sorted(unexpected)[:10]:
            print(f"    - {key}")
        if len(unexpected) > 10:
            print(f"    ... and {len(unexpected) - 10} more")

    if not missing and not unexpected:
        print("✓ Checkpoint loaded successfully (all keys matched)")

    # Prepare test input
    print("\n[4/5] Preparing test input...")
    test_text = "The quick brown fox jumps over the lazy dog"

    # HuggingFace tokenization
    hf_inputs = tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = hf_inputs["input_ids"]

    print(f"  Input text: '{test_text}'")
    print(f"  Token IDs shape: {input_ids.shape}")
    print(f"  Token IDs: {input_ids[0].tolist()}")

    # Run inference
    print("\n[5/5] Running inference and comparing outputs...")

    with torch.no_grad():
        # HuggingFace forward pass
        print("  Running HuggingFace model...")
        hf_outputs = hf_model(input_ids, use_cache=True)
        hf_logits = hf_outputs.logits

        # fairseq2 forward pass
        print("  Running fairseq2 model...")
        # Create batch layout
        seq_lens = [input_ids.shape[1]]
        batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
        fs2_logits = fs2_model(input_ids, batch_layout)

    # Compare outputs
    print("\n" + "="*80)
    print("PARITY RESULTS")
    print("="*80)

    print(f"\nOutput shapes:")
    print(f"  HF:  {hf_logits.shape}")
    print(f"  FS2: {fs2_logits.shape}")

    if hf_logits.shape != fs2_logits.shape:
        print("\n❌ SHAPE MISMATCH")
        return

    # Compute differences
    abs_diff = torch.abs(hf_logits - fs2_logits)
    rel_diff = abs_diff / (torch.abs(hf_logits) + 1e-8)

    print(f"\nAbsolute difference:")
    print(f"  Max:  {abs_diff.max().item():.6e}")
    print(f"  Mean: {abs_diff.mean().item():.6e}")
    print(f"  Std:  {abs_diff.std().item():.6e}")

    print(f"\nRelative difference:")
    print(f"  Max:  {rel_diff.max().item():.6e}")
    print(f"  Mean: {rel_diff.mean().item():.6e}")
    print(f"  Std:  {rel_diff.std().item():.6e}")

    # Check token predictions
    hf_predicted_ids = hf_logits.argmax(dim=-1)
    fs2_predicted_ids = fs2_logits.argmax(dim=-1)

    token_match = (hf_predicted_ids == fs2_predicted_ids).float().mean().item()
    print(f"\nToken prediction agreement: {token_match*100:.2f}%")

    # Detailed comparison of first few tokens
    print(f"\nFirst 5 token predictions:")
    print(f"  Position | HF Token | FS2 Token | Match | HF Prob | FS2 Prob")
    print(f"  " + "-"*70)

    hf_probs = torch.softmax(hf_logits, dim=-1)
    fs2_probs = torch.softmax(fs2_logits, dim=-1)

    for i in range(min(5, input_ids.shape[1])):
        hf_tok = hf_predicted_ids[0, i].item()
        fs2_tok = fs2_predicted_ids[0, i].item()
        hf_prob = hf_probs[0, i, hf_tok].item()
        fs2_prob = fs2_probs[0, i, fs2_tok].item()
        match = "✓" if hf_tok == fs2_tok else "✗"

        print(f"  {i:8d} | {hf_tok:8d} | {fs2_tok:9d} | {match:5s} | {hf_prob:7.4f} | {fs2_prob:8.4f}")

    # Determine pass/fail
    print("\n" + "="*80)
    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()

    # Thresholds for numerical parity
    ABS_THRESHOLD = 1e-3  # 0.001 absolute difference
    REL_THRESHOLD = 1e-2  # 1% relative difference

    if max_abs_diff < ABS_THRESHOLD and max_rel_diff < REL_THRESHOLD:
        print("✅ PARITY TEST PASSED")
        print(f"   Max absolute diff ({max_abs_diff:.6e}) < {ABS_THRESHOLD}")
        print(f"   Max relative diff ({max_rel_diff:.6e}) < {REL_THRESHOLD}")
    elif token_match > 0.99:
        print("⚠️  PARITY TEST PARTIAL PASS")
        print(f"   Token predictions match: {token_match*100:.2f}%")
        print(f"   But numerical differences exceed thresholds:")
        print(f"     Max absolute diff: {max_abs_diff:.6e} (threshold: {ABS_THRESHOLD})")
        print(f"     Max relative diff: {max_rel_diff:.6e} (threshold: {REL_THRESHOLD})")
    else:
        print("❌ PARITY TEST FAILED")
        print(f"   Max absolute diff: {max_abs_diff:.6e}")
        print(f"   Max relative diff: {max_rel_diff:.6e}")
        print(f"   Token match rate: {token_match*100:.2f}%")

    print("="*80)


if __name__ == "__main__":
    main()
