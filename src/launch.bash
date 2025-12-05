#!/bin/bash

export CUDA_VISIBLE_DEVICES=$(seq -s, $i $j)
echo 'Visible device ids:'
echo $CUDA_VISIBLE_DEVICES
echo 'Running script...'
torchrun --nproc_per_node $((j+1)) hg_load_test.py
