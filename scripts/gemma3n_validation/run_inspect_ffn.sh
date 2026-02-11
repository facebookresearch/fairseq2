#!/bin/bash
#SBATCH --job-name=inspect_ffn
#SBATCH --output=scripts/gemma3n_validation/inspect_ffn_%j.log
#SBATCH --partition=h100
#SBATCH --mem=50G
#SBATCH --time=0:10:00

set -e

echo "=== Inspecting Gemma3n FFN Dimensions ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo ""

/home/aerben/repos/fairseq2/.venv/bin/python3 scripts/gemma3n_validation/inspect_ffn_dims.py
