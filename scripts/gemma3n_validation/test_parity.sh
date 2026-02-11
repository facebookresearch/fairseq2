#!/bin/bash
set -e

VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Gemma3n Full Parity Test ==="
echo ""
echo "This test will:"
echo "  1. Load HuggingFace Gemma3n-E2B model"
echo "  2. Create fairseq2 Gemma3n model"
echo "  3. Convert HF checkpoint to fairseq2 format"
echo "  4. Run same input through both models"
echo "  5. Compare outputs for numerical parity"
echo ""
echo "Running on device: $(hostname)"
echo ""

$PYTHON /home/aerben/repos/fairseq2/scripts/gemma3n_validation/test_parity.py
