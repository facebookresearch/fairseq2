#!/bin/bash
# Qwen 3.5 2B training on FineWeb-Edu 10BT (1-node, all optimizations)
#SBATCH --job-name=qwen35_2B_1node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=168:00:00
#SBATCH --account=seamless_fs2
#SBATCH --qos=h200_lowest

#SBATCH --output=/checkpoint/smallomnillm/yunchaoyang1/qwen35_pretrain/2B/slurm_%j.out
#SBATCH --error=/checkpoint/smallomnillm/yunchaoyang1/qwen35_pretrain/2B/slurm_%j.err

OUTPUT_DIR=/checkpoint/smallomnillm/yunchaoyang1/qwen35_pretrain/2B/ablation_all_1node

mkdir -p "${OUTPUT_DIR}"

source ~/envs/fs081-pt290-cu128/bin/activate
cd /storage/home/yunchaoyang1/fairseq2

torchrun --standalone --nproc_per_node=8 -m recipes.lm.train \
  --config-file recipes/lm/train/configs/qwen35_2b_fineweb_edu_10bt.yaml \
  --config \
    "set:model.compile=true" \
    "set:model.compile_options.mode=max-autotune" \
    "set:dataset.max_num_tokens=32768" \
    "set:dataset.prefetch=8" \
    "set:trainer.activation_checkpointing.mode=none" \
    "set:trainer.fsdp.fp32_reduce=false" \
    "set:trainer.gc_every_n_steps=5000" \
  "${OUTPUT_DIR}"
