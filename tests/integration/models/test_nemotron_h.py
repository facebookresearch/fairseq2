# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration parity test for NemotronH.

Compares logits between HuggingFace and fairseq2 implementations.
Uses two-pass loading (HF first → free → FS2 → compare) due to model size.

Requires:
- GPU with 80GB+ VRAM (H200)
- mamba-ssm, causal-conv1d installed
- Checkpoint at /engshare/yunchaoyang1/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16

Run:
    srun --qos=h200_dev --partition=h200 --gpus=1 --cpus-per-task=24 \
      --account=smallomnillm --time=4:00:00 --pty bash
    conda activate fs2-090dev0-pt290-cu128
    cd /storage/home/yunchaoyang1/fairseq2-nemotron
    python tests/integration/models/test_nemotron_h.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Standard test prompts
TEST_PROMPTS = [
    "The capital of France is",
    "In machine learning, a neural network",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The theory of relativity states that",
    "Once upon a time in a land far away",
    "import torch\nimport torch.nn as nn\n\nclass",
    "The key ingredients for making chocolate cake are",
    "According to quantum mechanics, particles can",
    "In the year 2050, scientists discovered",
    "The meaning of life, the universe, and everything is",
]

MODEL_PATH = "/engshare/yunchaoyang1/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
RESULTS_DIR = Path("/storage/home/yunchaoyang1/ProfAI/projects/12_NemotronH_fairseq2_Implementation/experiments")


def get_hf_logits(
    model_path: str,
    prompts: list[str],
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Load HF model, compute logits on prompts, return logits dict."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HF model from {model_path} (dtype={dtype})...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="cuda:0",
    )
    model.eval()

    t1 = time.time()
    print(f"HF model loaded in {t1 - t0:.1f}s")
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    results = {}
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            outputs = model(**inputs)
        # Store logits on CPU
        results[prompt] = outputs.logits.cpu().float()
        print(f"  Prompt {i+1}/{len(prompts)}: {prompt[:40]}... shape={outputs.logits.shape}")

    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"HF model freed. GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    return results


def get_fs2_logits(
    model_path: str,
    prompts: list[str],
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Load FS2 model, compute logits on prompts, return logits dict."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from fairseq2.models.nemotron.config import NemotronHConfig
    from fairseq2.models.nemotron.factory import NemotronHFactory
    from fairseq2.models.nemotron.interop import convert_nemotron_h_state_dict
    from fairseq2.nn import BatchLayout

    print("Creating FS2 NemotronH model...")
    t0 = time.time()

    config = NemotronHConfig()
    factory = NemotronHFactory(config)

    # Create model in dtype
    with torch.device("meta"):
        model = factory.create_model()

    # Load and convert state dict
    print("Loading safetensors checkpoint...")
    model_path_p = Path(model_path)
    safetensor_files = sorted(model_path_p.glob("*.safetensors"))
    print(f"Found {len(safetensor_files)} safetensor files")

    hf_state_dict: dict[str, object] = {}
    for sf_file in safetensor_files:
        shard = load_file(str(sf_file), device="cpu")
        hf_state_dict.update(shard)

    print(f"Total HF keys: {len(hf_state_dict)}")

    # Convert keys
    fs2_state_dict = convert_nemotron_h_state_dict(hf_state_dict, config)
    print(f"Total FS2 keys: {len(fs2_state_dict)}")

    # Load into model
    # Cast tensors to the right dtype
    for key in fs2_state_dict:
        t = fs2_state_dict[key]
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            fs2_state_dict[key] = t.to(dtype)

    model.load_state_dict(fs2_state_dict, assign=True)  # type: ignore
    model = model.to("cuda:0")
    model.eval()

    t1 = time.time()
    print(f"FS2 model loaded in {t1 - t0:.1f}s")
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Tokenizer from HF
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    results = {}
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda:0")
        layout = BatchLayout.of(input_ids)

        with torch.no_grad():
            logits = model(input_ids, layout)

        results[prompt] = logits.cpu().float()
        print(f"  Prompt {i+1}/{len(prompts)}: {prompt[:40]}... shape={logits.shape}")

    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"FS2 model freed. GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    return results


def compare_logits(
    hf_results: dict[str, torch.Tensor],
    fs2_results: dict[str, torch.Tensor],
) -> dict[str, object]:
    """Compare logits between HF and FS2 implementations."""
    metrics: dict[str, object] = {}

    cos_sims = []
    top1_agreements = []

    for prompt in hf_results:
        hf_logits = hf_results[prompt]  # [1, seq_len, vocab]
        fs2_logits = fs2_results[prompt]  # [1, seq_len, vocab]

        assert hf_logits.shape == fs2_logits.shape, (
            f"Shape mismatch: HF={hf_logits.shape}, FS2={fs2_logits.shape}"
        )

        # Flatten for cosine similarity
        hf_flat = hf_logits.view(-1)
        fs2_flat = fs2_logits.view(-1)

        cos_sim = F.cosine_similarity(
            hf_flat.unsqueeze(0), fs2_flat.unsqueeze(0)
        ).item()
        cos_sims.append(cos_sim)

        # Top-1 token agreement
        hf_top1 = hf_logits.argmax(dim=-1)  # [1, seq_len]
        fs2_top1 = fs2_logits.argmax(dim=-1)  # [1, seq_len]
        agreement = (hf_top1 == fs2_top1).float().mean().item()
        top1_agreements.append(agreement)

        # Max absolute difference
        max_diff = (hf_logits - fs2_logits).abs().max().item()

        print(
            f"  {prompt[:30]:30s}  cos_sim={cos_sim:.6f}  "
            f"top1_agree={agreement:.3f}  max_diff={max_diff:.4f}"
        )

    avg_cos = sum(cos_sims) / len(cos_sims)
    min_cos = min(cos_sims)
    avg_top1 = sum(top1_agreements) / len(top1_agreements)

    metrics["cos_sim_mean"] = avg_cos
    metrics["cos_sim_min"] = min_cos
    metrics["top1_agreement_mean"] = avg_top1
    metrics["num_prompts"] = len(cos_sims)

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Cosine Similarity — Mean: {avg_cos:.6f}, Min: {min_cos:.6f}")
    print(f"  Top-1 Agreement  — Mean: {avg_top1:.3f}")
    print(f"  Threshold: cos_mean >= 0.999, cos_min >= 0.998, top1 >= 0.95")

    # Check thresholds
    passed = True
    if avg_cos < 0.999:
        print(f"  FAIL: cos_mean {avg_cos:.6f} < 0.999")
        passed = False
    if min_cos < 0.998:
        print(f"  FAIL: cos_min {min_cos:.6f} < 0.998")
        passed = False
    if avg_top1 < 0.95:
        print(f"  FAIL: top1 {avg_top1:.3f} < 0.95")
        passed = False

    if passed:
        print("  ✓ ALL PARITY CHECKS PASSED")
    else:
        print("  ✗ PARITY CHECK FAILED")

    metrics["passed"] = passed
    return metrics


def main() -> None:
    print("=" * 60)
    print("NemotronH Logit Parity Test")
    print("=" * 60)

    # Check prerequisites
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run on a GPU node.")
        sys.exit(1)

    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Download with: huggingface-cli download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")
        sys.exit(1)

    dtype = torch.bfloat16
    print(f"\nDtype: {dtype}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print()

    # Pass 1: HF logits
    print("=" * 60)
    print("PASS 1: HuggingFace Model")
    print("=" * 60)
    hf_results = get_hf_logits(MODEL_PATH, TEST_PROMPTS, dtype)

    # Pass 2: FS2 logits
    print()
    print("=" * 60)
    print("PASS 2: fairseq2 Model")
    print("=" * 60)
    fs2_results = get_fs2_logits(MODEL_PATH, TEST_PROMPTS, dtype)

    # Compare
    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    metrics = compare_logits(hf_results, fs2_results)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "parity_results.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
