#!/bin/bash
# Ablation: ALL optimizations combined
# compile + flash + max_num_tokens=32768 + no AC + bf16 reduce + gc=5000 + prefetch=8
#SBATCH --job-name=qwen35_ablation_all_4node
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=168:00:00
#SBATCH --account=smallomnillm
#SBATCH --qos=h200_smallomnillm_high

#SBATCH --output=/checkpoint/smallomnillm/yunchaoyang1/qwen35_pretrain/slurm_%j.out
#SBATCH --error=/checkpoint/smallomnillm/yunchaoyang1/qwen35_pretrain/slurm_%j.err

OUTPUT_DIR=/checkpoint/smallomnillm/yunchaoyang1/qwen35_pretrain/ablation_all_4node

mkdir -p "${OUTPUT_DIR}"

source ~/envs/fs081-pt290-cu128/bin/activate
cd /storage/home/yunchaoyang1/fairseq2

srun torchrun --nproc_per_node=8 \
  --nnodes=${SLURM_NNODES} \
  --rdzv_id=${SLURM_JOB_ID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$(scontrol show hostname $SLURM_NODELIST | head -n1):29500 \
  -m recipes.lm.train \
  --config-file recipes/lm/train/configs/qwen35_0.8b_fineweb_edu_10bt.yaml \
  --config \
    "set:model.compile=true" \
    "set:model.compile_options.mode=max-autotune" \
    "set:common.torch.default_sdpa=flash" \
    "set:dataset.max_num_tokens=32768" \
    "set:dataset.prefetch=8" \
    "set:trainer.activation_checkpointing.mode=none" \
    "set:trainer.fsdp.fp32_reduce=false" \
    "set:trainer.gc_every_n_steps=5000" \
  "${OUTPUT_DIR}"
