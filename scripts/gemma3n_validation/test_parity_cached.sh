#!/bin/bash
set -e

VENV=/home/aerben/repos/fairseq2/.venv
PYTHON=$VENV/bin/python

echo "=== Gemma3n Parity Test (Cached Model) ==="
echo ""
echo "Running on: $(hostname)"
echo ""

$PYTHON /home/aerben/repos/fairseq2/scripts/gemma3n_validation/test_parity_cached.py
