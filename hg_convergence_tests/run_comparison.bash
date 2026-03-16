#!/bin/bash

#SBATCH --job-name=convergence-comparison
#SBATCH --output=$HOME/fairseq2/hg_hardware_test/slurm_logs/comparison/comparison-%j.out
#SBATCH --error=$HOME/fairseq2/hg_hardware_test/slurm_logs/comparison/comparison-%j.err
#SBATCH --gpus 8
#SBATCH --account seamless_fs2
#SBATCH --qos h100_dev
#SBATCH --time 1409
#SBATCH --mem 512GB

source $HOME/miniconda3/bin/activate
# Replace with your env if needed
conda activate fs2_hg

set -euo pipefail

# Change to project directory first
# Replace with your fairseq2 dir if needed
cd $HOME/fairseq2/hg_hardware_test

# Orchestration script for parallel convergence testing
# Runs fairseq2 and HuggingFace training in parallel, then compares checkpoints

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
SCRIPT_DIR="$PROJECT_ROOT/hg_convergence_tests"

# Define directories
FAIRSEQ2_DIR="/checkpoint/seamless/richardyue/fs2_hg/comparison/fairseq2_checkpoints"
HF_DIR="/checkpoint/seamless/richardyue/fs2_hg/comparison/hf_checkpoints"
EXTRACTED_DATA_DIR="/checkpoint/seamless/richardyue/fs2_hg/comparison/extracted_batches"

# Step 1: Setup directories
echo "=== Step 1: Setting up directories ==="
mkdir -p "$PROJECT_ROOT/slurm_logs"
mkdir -p "$FAIRSEQ2_DIR"
mkdir -p "$HF_DIR"
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
        --num-batches 200

    if [ $? -ne 0 ]; then
        echo "ERROR: Dataset extraction failed"
        exit 1
    fi
fi

# Step 3: Run training in parallel on separate GPUs
echo "=== Step 3: Running training in parallel on 8 GPUs (4 per framework) ==="

# Get GPU device IDs from Slurm
if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    # Parse Slurm GPU allocation
    IFS=',' read -ra GPU_ARRAY <<< "$SLURM_JOB_GPUS"
    GPU_COUNT=${#GPU_ARRAY[@]}

    if [ $GPU_COUNT -lt 8 ]; then
        echo "Error: Need 8 GPUs but only $GPU_COUNT allocated"
        exit 1
    fi

    # Split GPUs: first 4 for fairseq2, last 4 for HF
    FAIRSEQ2_GPUS="${GPU_ARRAY[0]},${GPU_ARRAY[1]},${GPU_ARRAY[2]},${GPU_ARRAY[3]}"
    HF_GPUS="${GPU_ARRAY[4]},${GPU_ARRAY[5]},${GPU_ARRAY[6]},${GPU_ARRAY[7]}"
else
    # Fallback for non-Slurm environments
    FAIRSEQ2_GPUS="0,1,2,3"
    HF_GPUS="4,5,6,7"
fi

echo "fairseq2 GPUs: $FAIRSEQ2_GPUS"
echo "HuggingFace GPUs: $HF_GPUS"

# Launch fairseq2 training on first 4 GPUs in background
echo "Starting fairseq2 training..."
CUDA_VISIBLE_DEVICES=$FAIRSEQ2_GPUS torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    "$SCRIPT_DIR/train_fairseq2_from_batches.py" \
    --data-dir "$EXTRACTED_DATA_DIR" \
    --output-dir "$FAIRSEQ2_DIR" \
    --num-steps 10000 \
    --seed 2 \
    > "$FAIRSEQ2_DIR/training.log" 2>&1 &
FAIRSEQ2_PID=$!

# Check if transformers is available
if python -c "import transformers" 2>/dev/null; then
    # Launch HuggingFace training on last 4 GPUs in background
    echo "Starting HuggingFace training..."
    CUDA_VISIBLE_DEVICES=$HF_GPUS torchrun \
        --nproc_per_node=4 \
        --master_port=29501 \
        "$SCRIPT_DIR/train_hf_multi_gpu.py" \
        --data-dir "$EXTRACTED_DATA_DIR" \
        --output-dir "$HF_DIR" \
        --num-steps 10000 \
        --seed 2 \
        > "$HF_DIR/training.log" 2>&1 &
    HF_PID=$!

    # Wait for both processes to complete
    echo "Waiting for both training processes to complete..."
    wait $FAIRSEQ2_PID
    FAIRSEQ2_EXIT_CODE=$?
    echo "fairseq2 training completed with exit code: $FAIRSEQ2_EXIT_CODE"

    wait $HF_PID
    HF_EXIT_CODE=$?
    echo "HuggingFace training completed with exit code: $HF_EXIT_CODE"
else
    # Only wait for fairseq2 if HF is not available
    echo "WARNING: transformers not installed, skipping HuggingFace training"
    wait $FAIRSEQ2_PID
    FAIRSEQ2_EXIT_CODE=$?
    echo "fairseq2 training completed with exit code: $FAIRSEQ2_EXIT_CODE"
    HF_EXIT_CODE=0
fi

# Step 4: Check training results
echo ""
echo "=== Step 4: Training Results ==="
echo "fairseq2 exit code: $FAIRSEQ2_EXIT_CODE"
if python -c "import transformers" 2>/dev/null; then
    echo "HuggingFace exit code: $HF_EXIT_CODE"
else
    echo "HF: skipped (not installed)"
fi

if [ $FAIRSEQ2_EXIT_CODE -ne 0 ]; then
    echo "ERROR: fairseq2 training failed with exit code $FAIRSEQ2_EXIT_CODE"
    echo "Check logs at: $FAIRSEQ2_DIR/training.log"
    exit 1
fi

if python -c "import transformers" 2>/dev/null && [ $HF_EXIT_CODE -ne 0 ]; then
    echo "ERROR: HuggingFace training failed with exit code $HF_EXIT_CODE"
    echo "Check logs at: $HF_DIR/training.log"
    exit 1
fi

echo "Training completed successfully!"

# Step 5: Compare checkpoints (only if HuggingFace ran)
if python -c "import transformers" 2>/dev/null; then
    echo ""
    echo "=== Step 5: Comparing checkpoints ==="
    python "$SCRIPT_DIR/compare_checkpoints_simple.py" \
        --fairseq2-checkpoint "$FAIRSEQ2_DIR/checkpoint_10000.pt" \
        --hf-checkpoint "$HF_DIR/checkpoint_10000.pt"

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
    echo "HuggingFace training was not run (not installed)"
    echo ""
    echo "=== SUCCESS: fairseq2 training completed ==="
    echo "To run convergence comparison, install transformers and re-run"
    exit 0
fi
