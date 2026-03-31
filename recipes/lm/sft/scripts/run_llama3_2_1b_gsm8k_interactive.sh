#!/bin/bash
# Run FA3 benchmark directly on a GPU node (no sbatch needed).
# Usage: srun --gpus-per-node=8 --ntasks-per-node=1 --cpus-per-task=96 bash recipes/lm/sft/scripts/run_llama3_2_1b_gsm8k_interactive.sh

set -e

BASE_OUTPUT_DIR=/checkpoint/seamless/yunchaoyang1/llama_sdpa_bench
NUM_STEPS=100
CONFIG_FILE=recipes/lm/sft/configs/llama3_2_1b_instruct_gsm8k.yaml

cd /storage/home/yunchaoyang1/fairseq2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate smallomni-2026-02-06

echo "Host: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

run_bench() {
  local name=$1
  local sdpa=$2
  local extra_config=$3

  echo "========== ${name} =========="
  local start=$(date +%s)

  torchrun --standalone --nproc_per_node=8 -m recipes.lm.sft \
    --config-file "${CONFIG_FILE}" \
    "${BASE_OUTPUT_DIR}/${name}" \
    --config common.torch.default_sdpa="${sdpa}" regime.num_steps="${NUM_STEPS}" ${extra_config}

  local end=$(date +%s)
  local elapsed=$((end - start))
  echo "  → ${name} finished in ${elapsed}s"
  echo ""

  # Store timing
  echo "${name} ${elapsed}" >> /tmp/sdpa_bench_times.txt
}

rm -f /tmp/sdpa_bench_times.txt

run_bench "flash3"      "flash3"      "dataset.packing=true"
run_bench "flash2"      "flash2"      "dataset.packing=true"
run_bench "torch_flash" "torch_flash" ""
run_bench "torch"       "torch"       ""

echo ""
echo "=========================================="
echo "  SDPA Benchmark Summary (${NUM_STEPS} steps)"
echo "=========================================="
echo ""
printf "%-15s | %-8s | %-12s\n" "SDPA" "Packing" "Wall Time(s)"
printf "%-15s-+-%-8s-+-%-12s\n" "---------------" "--------" "------------"

while read -r name elapsed; do
  case ${name} in
    flash3|flash2) packing="packed" ;;
    *) packing="padded" ;;
  esac
  printf "%-15s | %-8s | %-12s\n" "${name}" "${packing}" "${elapsed}"
done < /tmp/sdpa_bench_times.txt

echo ""
echo "Detailed logs: ${BASE_OUTPUT_DIR}/<variant>/"
echo "Done."
