#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Gemma4 31B SFT on FineWeb-Edu 10BT (8x H200, 1 node)
#
# Validated: 200 steps, NLL 1.95 -> 2.07, peak mem 81.70 GiB (59%).
#
# Usage:
#   sbatch recipes/lm/sft/scripts/run_gemma4_31b_fineweb.sh

#SBATCH --job-name=gemma4_31b_sft
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

OUTPUT_DIR="${OUTPUT_DIR:-/checkpoint/${USER}/gemma4/sft_31b_fineweb}"

mkdir -p "${OUTPUT_DIR}"

cd "$(dirname "$0")/../../../.."

torchrun --standalone --nproc_per_node=8 -m recipes.lm.sft \
  --config-file recipes/lm/sft/configs/gemma4_31b_fineweb.yaml \
  "${OUTPUT_DIR}"
