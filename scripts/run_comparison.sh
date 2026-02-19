#!/bin/bash

#SBATCH --job-name=convergence-comparison
#SBATCH --output=/home/richardyue/fairseq2/hg_hardware_test/slurm_logs/comparison-%j.out
#SBATCH --error=/home/richardyue/fairseq2/hg_hardware_test/slurm_logs/comparison-%j.err
#SBATCH --gpus 8
#SBATCH --account seamless_fs2
#SBATCH --qos h100_dev
#SBATCH --time 1409
#SBATCH --mem 128GB

source ~/miniconda3/bin/activate
source activate fs2v080

set -euo pipefail

# Change to project directory first
cd /home/richardyue/fairseq2/hg_hardware_test

# Orchestration script for parallel convergence testing
# Runs fairseq2 and Unsloth training in parallel, then compares checkpoints

# Parse command-line arguments
SKIP_EXTRACTION=false
for arg in "$@"; do
    case $arg in
        --skip-extraction)
            SKIP_EXTRACTION=true
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--skip-extraction]"
            exit 1
            ;;
    esac
done

# Set project root and script directory (we already cd'd to project root above)
PROJECT_ROOT="/home/richardyue/fairseq2/hg_hardware_test"
SCRIPT_DIR="$PROJECT_ROOT/scripts"

# Define directories
FAIRSEQ2_DIR="$PROJECT_ROOT/out/fairseq2_checkpoints"
UNSLOTH_DIR="$PROJECT_ROOT/out/unsloth_checkpoints"
EXTRACTED_DATA_DIR="$PROJECT_ROOT/out/extracted_batches"

# Step 1: Setup directories
echo "=== Step 1: Setting up directories ==="
mkdir -p "$PROJECT_ROOT/slurm_logs"
mkdir -p "$FAIRSEQ2_DIR"
mkdir -p "$UNSLOTH_DIR"
mkdir -p "$EXTRACTED_DATA_DIR"

# Step 2: Extract dataset batches (unless skipped)
if [ "$SKIP_EXTRACTION" = true ]; then
    echo "=== Step 2: Skipping extraction (--skip-extraction flag set) ==="
    # Verify extracted data exists
    if [ ! -d "$EXTRACTED_DATA_DIR" ] || [ -z "$(ls -A "$EXTRACTED_DATA_DIR")" ]; then
        echo "ERROR: --skip-extraction specified but no extracted data found in $EXTRACTED_DATA_DIR"
        exit 1
    fi
    echo "Using existing extracted data from $EXTRACTED_DATA_DIR"
else
    echo "=== Step 2: Extracting dataset batches ==="
    python "$SCRIPT_DIR/extract_fairseq2_batches.py" \
        --output-dir "$EXTRACTED_DATA_DIR" \
        --num-batches 10

    if [ $? -ne 0 ]; then
        echo "ERROR: Dataset extraction failed"
        exit 1
    fi
fi

# Step 3: Run parallel training
echo "=== Step 3: Running parallel training ==="

# Launch fairseq2 training on GPUs 0-3
echo "Starting fairseq2 training on GPUs 0-3..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m recipes.lm.sft \
    --config-file recipes/lm/sft/configs/gemma_3_2b_it_gsm8k.yaml \
    --output-dir "$FAIRSEQ2_DIR" \
    > "$FAIRSEQ2_DIR/training.log" 2>&1 &
FAIRSEQ2_PID=$!
echo "fairseq2 training started (PID: $FAIRSEQ2_PID)"

# Check if unsloth is available
if python -c "import unsloth" 2>/dev/null; then
    # Launch Unsloth training on GPUs 4-7
    echo "Starting Unsloth training on GPUs 4-7..."
    CUDA_VISIBLE_DEVICES=4,5,6,7 python "$SCRIPT_DIR/train_unsloth.py" \
        --extracted-data-dir "$EXTRACTED_DATA_DIR" \
        --output-dir "$UNSLOTH_DIR" \
        --num-batches 10 \
        > "$UNSLOTH_DIR/training.log" 2>&1 &
    UNSLOTH_PID=$!
    echo "Unsloth training started (PID: $UNSLOTH_PID)"
else
    echo "WARNING: unsloth not installed, skipping Unsloth training"
    UNSLOTH_PID=""
fi

# Step 4: Wait for both training runs to complete
echo "=== Step 4: Waiting for training completion ==="
FAIRSEQ2_EXIT_CODE=0
UNSLOTH_EXIT_CODE=0

# Wait for fairseq2
echo "Waiting for fairseq2 training (PID: $FAIRSEQ2_PID)..."
wait $FAIRSEQ2_PID || FAIRSEQ2_EXIT_CODE=$?

# Wait for Unsloth (if it was started)
if [ -n "$UNSLOTH_PID" ]; then
    echo "Waiting for Unsloth training (PID: $UNSLOTH_PID)..."
    wait $UNSLOTH_PID || UNSLOTH_EXIT_CODE=$?
else
    echo "Skipping Unsloth wait (not started)"
fi

# Check exit codes
echo ""
echo "=== Training Results ==="
echo "fairseq2 exit code: $FAIRSEQ2_EXIT_CODE"
if [ -n "$UNSLOTH_PID" ]; then
    echo "Unsloth exit code: $UNSLOTH_EXIT_CODE"
else
    echo "Unsloth: skipped (not installed)"
fi

if [ $FAIRSEQ2_EXIT_CODE -ne 0 ]; then
    echo "ERROR: fairseq2 training failed with exit code $FAIRSEQ2_EXIT_CODE"
    echo "Check logs at: $FAIRSEQ2_DIR/training.log"
    exit 1
fi

if [ -n "$UNSLOTH_PID" ] && [ $UNSLOTH_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Unsloth training failed with exit code $UNSLOTH_EXIT_CODE"
    echo "Check logs at: $UNSLOTH_DIR/training.log"
    exit 1
fi

echo "Training completed successfully!"

# Step 5: Compare checkpoints (only if Unsloth ran)
if [ -n "$UNSLOTH_PID" ]; then
    echo ""
    echo "=== Step 5: Comparing checkpoints ==="
    python "$SCRIPT_DIR/compare_checkpoints.py" \
        --fairseq2-dir "$FAIRSEQ2_DIR" \
        --unsloth-dir "$UNSLOTH_DIR" \
        --num-batches 10

    COMPARISON_EXIT_CODE=$?

    if [ $COMPARISON_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=== SUCCESS: Convergence test passed! ==="
        echo "Checkpoints are identical within tolerance."
        exit 0
    else
        echo ""
        echo "=== FAILURE: Convergence test failed ==="
        echo "Checkpoints differ beyond acceptable tolerance."
        exit 1
    fi
else
    echo ""
    echo "=== Step 5: Skipping checkpoint comparison ==="
    echo "Unsloth was not run (not installed)"
    echo ""
    echo "=== SUCCESS: fairseq2 training completed ==="
    echo "To run convergence comparison, install unsloth and re-run"
    exit 0
fi
