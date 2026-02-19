#!/bin/bash
#SBATCH --job-name=extract-batches
#SBATCH --output=/checkpoint/seamless/richardyue/extract-%j.out
#SBATCH --error=/checkpoint/seamless/richardyue/extract-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

# Exit on error
set -e

echo "=========================================="
echo "Extracting fairseq2 Batches for Convergence Test"
echo "=========================================="

# Configuration
EXTRACTED_DATA="/checkpoint/seamless/richardyue/extracted_data"
NUM_BATCHES=100

# Create output directory
mkdir -p "$EXTRACTED_DATA"

# Run extraction
python scripts/extract_fairseq2_batches.py \
    --output-dir "$EXTRACTED_DATA" \
    --num-batches $NUM_BATCHES \
    --tokenizer google/gemma-3-1b-it \
    --dataset-path hg://facebook/fairseq2-lm-gsm8k \
    --split sft_train \
    --max-seq-len 4096 \
    --seed 2

echo "=========================================="
echo "Extraction complete!"
echo "Saved $NUM_BATCHES batches to $EXTRACTED_DATA"
echo "=========================================="
