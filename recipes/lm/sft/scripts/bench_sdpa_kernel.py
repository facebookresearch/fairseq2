#!/usr/bin/env python3
"""Micro-benchmark: flash_attn_3 vs flash_attn_2 kernel-level comparison.

Directly times the attention kernels on synthetic data at various sequence
lengths, bypassing the full training loop so that only the SDPA compute is
measured.

Usage (on an interactive GPU node):
    python recipes/lm/sft/scripts/bench_sdpa_kernel.py
"""

import torch
import time
import statistics

from flash_attn import flash_attn_func as flash2_func  # type: ignore[import-untyped]
from fairseq2.models.transformer.sdpa.flash3 import flash_attn_3 as flash3_func


BATCH_SIZE = 2
NUM_HEADS = 32
HEAD_DIM = 128
SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768]
WARMUP = 5
REPEATS = 20
DTYPE = torch.bfloat16
DEVICE = "cuda"


def bench_kernel(fn, q, k, v):
    # warmup
    for _ in range(WARMUP):
        fn(q, k, v, causal=True)
    torch.cuda.synchronize()

    times = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(q, k, v, causal=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    return statistics.median(times)


def main():
    print(f"{'seq_len':>8}  {'flash2 (ms)':>12}  {'flash3 (ms)':>12}  {'speedup':>8}")
    print("-" * 48)

    for seq_len in SEQ_LENS:
        q = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        k = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        v = torch.randn(BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

        t2 = bench_kernel(flash2_func, q, k, v)
        t3 = bench_kernel(flash3_func, q, k, v)

        speedup = t2 / t3
        print(f"{seq_len:>8}  {t2:>12.3f}  {t3:>12.3f}  {speedup:>7.2f}x")

        del q, k, v
        torch.cuda.empty_cache()

    print()


if __name__ == "__main__":
    main()
