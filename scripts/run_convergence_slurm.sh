#!/bin/bash
#SBATCH --job-name=convergence-test
#SBATCH --output=/checkpoint/seamless/richardyue/convergence-%j.out
#SBATCH --error=/checkpoint/seamless/richardyue/convergence-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=8:00:00

# Exit on error
set -e

echo "=========================================="
echo "fairseq2 vs Unsloth Convergence Test (Slurm)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load any required modules (adjust for your cluster)
# module load cuda/12.1
# module load python/3.10

# Change to project directory
cd /home/richardyue/fairseq2/hg_hardware_test

# Run the convergence test
./scripts/run_comparison.sh

echo "=========================================="
echo "Convergence test complete!"
echo "=========================================="
