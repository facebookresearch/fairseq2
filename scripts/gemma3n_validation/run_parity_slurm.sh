#!/bin/bash
#SBATCH --job-name=gemma3n_parity
#SBATCH --output=/home/aerben/repos/fairseq2/scripts/gemma3n_validation/parity_test_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --partition=h100
#SBATCH --gpus-per-task=1

set -e

VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Gemma3n Parity Test on SLURM ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Run the cached parity test
$PYTHON /home/aerben/repos/fairseq2/scripts/gemma3n_validation/test_parity_cached.py

echo ""
echo "Parity test complete!"
