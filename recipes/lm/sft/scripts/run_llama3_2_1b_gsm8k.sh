#!/bin/bash
#SBATCH --job-name=llama3_2_1b_gsm8k_sft
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --output=log/slurm_%j.out
#SBATCH --error=log/slurm_%j.err
#SBATCH --account=seamless
#SBATCH --qos=h200_seamless_high

# SLURM wrapper — just calls the benchmark script.
# Submit: sbatch recipes/lm/sft/scripts/run_llama3_2_1b_gsm8k.sh

bash /storage/home/yunchaoyang1/fairseq2/recipes/lm/sft/scripts/bench_sdpa.sh
