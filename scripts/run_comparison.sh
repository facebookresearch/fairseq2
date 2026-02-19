#!/bin/bash
set -euo pipefail

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

# Get the absolute path to the project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Define directories
FAIRSEQ2_DIR="$PROJECT_ROOT/out/fairseq2_checkpoints"
UNSLOTH_DIR="$PROJECT_ROOT/out/unsloth_checkpoints"
EXTRACTED_DATA_DIR="$PROJECT_ROOT/out/extracted_batches"

# Step 1: Setup directories
echo "=== Step 1: Setting up directories ==="
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
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq2 lm gemma3_2b_it \
    --config-file recipes/lm/sft/configs/gemma_3_2b_it_gsm8k.yaml \
    --output-dir "$FAIRSEQ2_DIR" \
    > "$FAIRSEQ2_DIR/training.log" 2>&1 &
FAIRSEQ2_PID=$!
echo "fairseq2 training started (PID: $FAIRSEQ2_PID)"

# Launch Unsloth training on GPUs 4-7
echo "Starting Unsloth training on GPUs 4-7..."
CUDA_VISIBLE_DEVICES=4,5,6,7 python "$SCRIPT_DIR/train_unsloth.py" \
    --extracted-data-dir "$EXTRACTED_DATA_DIR" \
    --output-dir "$UNSLOTH_DIR" \
    --num-batches 10 \
    > "$UNSLOTH_DIR/training.log" 2>&1 &
UNSLOTH_PID=$!
echo "Unsloth training started (PID: $UNSLOTH_PID)"

# Step 4: Wait for both training runs to complete
echo "=== Step 4: Waiting for training completion ==="
FAIRSEQ2_EXIT_CODE=0
UNSLOTH_EXIT_CODE=0

# Wait for fairseq2
echo "Waiting for fairseq2 training (PID: $FAIRSEQ2_PID)..."
wait $FAIRSEQ2_PID || FAIRSEQ2_EXIT_CODE=$?

# Wait for Unsloth
echo "Waiting for Unsloth training (PID: $UNSLOTH_PID)..."
wait $UNSLOTH_PID || UNSLOTH_EXIT_CODE=$?

# Check exit codes
echo ""
echo "=== Training Results ==="
echo "fairseq2 exit code: $FAIRSEQ2_EXIT_CODE"
echo "Unsloth exit code: $UNSLOTH_EXIT_CODE"

if [ $FAIRSEQ2_EXIT_CODE -ne 0 ]; then
    echo "ERROR: fairseq2 training failed with exit code $FAIRSEQ2_EXIT_CODE"
    echo "Check logs at: $FAIRSEQ2_DIR/training.log"
    exit 1
fi

if [ $UNSLOTH_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Unsloth training failed with exit code $UNSLOTH_EXIT_CODE"
    echo "Check logs at: $UNSLOTH_DIR/training.log"
    exit 1
fi

echo "Both training runs completed successfully!"

# Step 5: Compare checkpoints
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
