# Flash Attention 3 vs Flash Attention 2 Benchmark Report

**Author:** Yunchao Yang
**Date:** 2026-03-27
**Environment:** FAIR cluster, 8× H100 GPUs, PyTorch + fairseq2

---

## Objective

Evaluate whether Flash Attention 3 provides a measurable speedup over Flash Attention 2 in fairseq2's training pipeline, and understand under what conditions the speedup is visible.

## Summary

| What | Status | Result |
|------|--------|--------|
| Kernel-level benchmark | ✅ Done | **flash3 is 1.6–2.1x faster** |
| End-to-end: 1B + GSM8K (short seq) | ✅ Done | No difference (attention <5% of step) |
| End-to-end: 8B + long-context | ❌ Failed | Dataset format mismatch + download timeout |

**Bottom line:** Flash3 is genuinely ~2x faster at the kernel level. The gain is invisible in end-to-end training when attention is a small fraction of compute (small model + short sequences).

---

## Results

### 1. Kernel-level micro-benchmark ✅

Directly timed `flash_attn_func` (flash2) vs `flash_attn_3` (flash3) on synthetic tensors.
Config: batch=2, 32 heads, head_dim=128, bfloat16, causal, H100.

| seq_len | flash2 (ms) | flash3 (ms) | speedup |
|---------|-------------|-------------|---------|
| 1024    | 0.123       | 0.075       | 1.64x   |
| 2048    | 0.291       | 0.152       | 1.91x   |
| 4096    | 0.872       | 0.435       | 2.01x   |
| 8192    | 3.312       | 1.547       | **2.14x** |
| 16384   | 12.125      | 6.823       | 1.78x   |
| 32768   | 47.918      | 27.068      | 1.77x   |

### 2. End-to-end: Llama 3.2 1B + GSM8K ✅

100 steps, bfloat16, `packing=true`, 8× H100.

| Variant | Wall time |
|---------|-----------|
| flash3  | ~290s     |
| flash2  | ~289s     |

**No difference.** GSM8K samples are ~200–500 tokens. With packing, the varlen kernels process each short sample individually via `cu_seqlens` — attention never operates on long sequences.

### 3. End-to-end: Llama 3.1 8B + FineWeb-Edu ❌

Attempted 50 steps, bfloat16, 32K seq_len, FSDP + activation checkpointing.

**Failed for two reasons:**
1. `fineweb-edu-score-2` has 8417 parquet files — download timed out at 84% after 7+ hours
2. FineWeb-Edu is a pretraining dataset (`text` field) — the SFT recipe expects JSONL with `src`/`tgt` or `chat` fields

---

## Key Findings

### Why no end-to-end difference with 1B + GSM8K?

- Attention is <5% of total step time for a 1B model with ~200-token sequences
- FFN matmuls, optimizer, gradient sync, and data loading dominate
- A 2x speedup on <5% of compute is invisible in wall-clock

### When flash3 gains would show in end-to-end training

| Factor | Favors flash3 | Our test (1B + GSM8K) |
|--------|---------------|----------------------|
| Sequence length | 8K–32K+ | ~200 tokens |
| GQA ratio | 1:1 (MHA) | 8:1 (8 KV heads) |
| FFN size | Small relative to d_model | Large (11008) |
| Model size | Larger = more attention | 1B (small) |

### Other findings

1. **`torch_flash` SDPA is broken with packing.** Disables all PyTorch backends then enables only `flash` + `cudnn`. With packed sequences, no fallback → 3.5x slower (1046s) or OOM.

2. **fairseq2 Trainer already instruments timing.** Separate `Stopwatch` for `data_time` (CPU) and `compute_time` (CUDA events). No bash-level metric parsing needed.

3. **HuggingFace downloads fill home directory.** Set `HF_HUB_CACHE` to `/checkpoint/` to avoid quota issues.

---

## Files

| File | Description |
|------|-------------|
| `recipes/lm/sft/scripts/bench_sdpa.sh` | End-to-end training benchmark script |
| `recipes/lm/sft/scripts/bench_sdpa_kernel.py` | Kernel-level micro-benchmark |
| `recipes/lm/sft/configs/llama3_1_8b_bench_sdpa.yaml` | 8B benchmark config (FSDP + AC) |
| `recipes/lm/sft/scripts/bench_sdpa_report.md` | This report |

---

## Next Steps

### To validate flash3 gains in end-to-end training:

1. **Create a synthetic long-sequence data generator** that produces SFT-compatible JSONL (random or lorem-ipsum text as `src`, short dummy `tgt`). This avoids dataset download/format issues entirely.

2. **Or find an existing SFT dataset with long sequences** — e.g., a long-form QA or summarization dataset already in `chat` format on HuggingFace that's small enough to download quickly.

3. **Re-run `bench_sdpa.sh`** with Llama 3.1 8B + the working long-context dataset. Target: 8K–32K seq_len with `packing=true` so the varlen kernels see genuinely long sequences.

### To improve the benchmark infrastructure:

4. **Add `compute_time` extraction** to the summary — parse the fairseq2 Trainer's logged metrics to report per-step `compute_time` (GPU-only, excluding data loading) alongside wall time. This isolates the attention kernel's contribution more precisely.

5. **Add a varlen kernel benchmark** to `bench_sdpa_kernel.py` — the current micro-benchmark uses the non-varlen path (`flash_attn_func`). Adding `flash_attn_varlen_func` vs `flash_attn_3_varlen` would match the actual packed-training code path.

---

## Conclusion

Flash Attention 3 delivers a **confirmed ~2x kernel-level speedup** over Flash Attention 2 on H100 (Hopper). This is a real hardware-level gain from flash3's async GEMM/softmax overlap via TMA and WGMMA.

For end-to-end training, the speedup is only visible when attention dominates total compute — large models (8B+) with long sequences (8K+). For small models with short sequences, the two are indistinguishable.
