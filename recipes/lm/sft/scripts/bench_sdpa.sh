#!/bin/bash
# Benchmark: Flash Attention 3 vs Flash Attention 2
#
# Runs end-to-end SFT training on Llama 3.1 8B with 32K sequences to measure
# the wall-clock impact of flash3 vs flash2 when attention is a significant
# fraction of total compute.
#
# Prerequisites:
#   - 8× H100 GPUs (interactive srun or sbatch)
#   - conda env with fairseq2 + flash-attn installed
#
# Usage:
#   bash recipes/lm/sft/scripts/bench_sdpa.sh 2>&1 | tee bench_sdpa.log

set -e

BASE_OUTPUT_DIR=/checkpoint/seamless/yunchaoyang1/llama_sdpa_bench
CONFIG_FILE=recipes/lm/sft/configs/llama3_1_8b_bench_sdpa.yaml
TIMES_FILE=/tmp/sdpa_bench_times.txt

cd /storage/home/yunchaoyang1/fairseq2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate smallomni-2026-02-06

export HF_HUB_CACHE=/checkpoint/seamless/yunchaoyang1/.cache/huggingface/hub

echo "========== Environment =========="
echo "Host:   $(hostname)"
echo "Date:   $(date)"
echo "GPUs:   $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)× $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Config: ${CONFIG_FILE}"
echo "Cache:  ${HF_HUB_CACHE}"
echo ""

> "${TIMES_FILE}"

run_bench() {
  local name=$1
  local sdpa=$2

  echo "========== ${name} =========="
  local t0=$SECONDS

  torchrun --standalone --nproc_per_node=8 -m recipes.lm.sft \
    --config-file "${CONFIG_FILE}" \
    "${BASE_OUTPUT_DIR}/${name}" \
    --config common.torch.default_sdpa="${sdpa}"

  local elapsed=$(( SECONDS - t0 ))
  echo "${name}  ${elapsed}s"
  echo "${name}  ${elapsed}s" >> "${TIMES_FILE}"
  echo ""
}

run_bench "flash3_32k" "flash3"
run_bench "flash2_32k" "flash2"

echo "========== Summary =========="
column -t "${TIMES_FILE}"
echo ""
echo "Detailed per-step metrics (compute_time, data_time, etc.) in:"
echo "  ${BASE_OUTPUT_DIR}/<variant>/"
