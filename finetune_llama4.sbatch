#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=llama4_sft
#SBATCH --nodes=4                # <-- number of nodes (set this)
#SBATCH --ntasks-per-node=8      # one task per GPU
#SBATCH --gres=gpu:8             # 8 GPUs per node
#SBATCH --cpus-per-task=12
#SBATCH --mem=0                  # use all available memory on node
#SBATCH --time=48:00:00          # walltime limit
#SBATCH --account=metaspeechssl
#SBATCH --qos=metaspeechssl_high
#SBATCH --output=slurm_logs/%x-%j.out  # log file pattern

echo "Job id is $SLURM_JOBID"
JOB_NAME_ID="${SLURM_JOB_NAME}_${SLURM_JOBID}"
echo "${JOB_NAME_ID}"

# --- Environment setup ---
source /fsx-checkpoints/mgleize/envs/fs2_ext_l4_cards/bin/activate 

# --- Run the job ---
srun python -m recipes.lm.sft \
  --config-file recipes/lm/sft/configs/llama4_scout_gsm8k_local.yaml \
  /zfs-home/mgleize/outputs/llama4_sft_tests/ \
  --config \
  common.metric_recorders.wandb.run_name=$JOB_NAME_ID

# --config-file recipes/lm/sft/configs/llama4_scout_gsm8k_local.yaml
# --config-file recipes/lm/sft/configs/llama3_2_1b_gsm8k.yaml \