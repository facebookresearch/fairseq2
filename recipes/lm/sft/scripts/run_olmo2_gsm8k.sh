#!/bin/bash
#SBATCH --job-name=olmo2_gsm8k_sft
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --output=/checkpoint/seamless/yunchaoyang1/olmo/slurm_%j.out
#SBATCH --error=/checkpoint/seamless/yunchaoyang1/olmo/slurm_%j.err

OUTPUT_DIR=/checkpoint/seamless/yunchaoyang1/olmo

mkdir -p "${OUTPUT_DIR}"

source /home/yunchaoyang1/envs/fs2-v0.8.0dev0-pt290-cu128/bin/activate
cd /storage/home/yunchaoyang1/fairseq2

torchrun --standalone --nproc_per_node=8 -m recipes.lm.sft \
  --config-file recipes/lm/sft/configs/olmo2_1b_gsm8k.yaml \
  "${OUTPUT_DIR}"
