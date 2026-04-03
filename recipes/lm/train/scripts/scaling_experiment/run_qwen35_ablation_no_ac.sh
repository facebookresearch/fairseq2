#!/bin/bash
# Ablation: Disable activation checkpointing
# Baseline → activation_checkpointing.mode=none
#SBATCH --job-name=qwen35_ablation_no_ac
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --account=smallomnillm
#SBATCH --qos=h200_smallomnillm_high

#SBATCH --output=/checkpoint/smallomnillm/yunchaoyang1/qwen35_pretrain/slurm_%j.out
#SBATCH --error=/checkpoint/smallomnillm/yunchaoyang1/qwen35_pretrain/slurm_%j.err

OUTPUT_DIR=/checkpoint/smallomnillm/yunchaoyang1/qwen35_pretrain/ablation_no_ac

mkdir -p "${OUTPUT_DIR}"

source ~/envs/fs081-pt290-cu128/bin/activate
cd /storage/home/yunchaoyang1/fairseq2

torchrun --standalone --nproc_per_node=8 -m recipes.lm.train \
  --config-file recipes/lm/train/configs/qwen35_0.8b_fineweb_edu_10bt.yaml \
  --config \
    "set:trainer.activation_checkpointing.mode=none" \
  "${OUTPUT_DIR}"
