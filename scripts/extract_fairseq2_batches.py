#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Extract batches from fairseq2's data pipeline for convergence testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fairseq2.data.tokenizers import load_tokenizer
from fairseq2.gang import setup_default_gang
from fairseq2.logging import get_log_writer

from recipes.lm.sft.dataset import (
    LMSFTDataset,
    LMSFTDataSource,
    DataReadOptions,
    StaticBatching,
)


log = get_log_writer(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract batches from fairseq2 data pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save extracted batches",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
        help="Number of batches to extract",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="google/gemma-3-1b-it",
        help="Tokenizer to use",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="hg://facebook/fairseq2-lm-gsm8k",
        help="Dataset path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="sft_train",
        help="Dataset split",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Random seed",
    )

    args = parser.parse_args()

    # Setup gang (for distributed data loading)
    gang = setup_default_gang()

    # Only rank 0 saves batches
    if gang.rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Extracting {args.num_batches} batches to {args.output_dir}")

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer, family="hg")
    log.info(f"Loaded tokenizer: {args.tokenizer}")

    # Create dataset
    sources = {
        "train": [
            LMSFTDataSource(
                path=args.dataset_path,
                split=args.split,
                weight=1.0,
            )
        ]
    }
    dataset = LMSFTDataset(sources)
    log.info(f"Created dataset from: {args.dataset_path}")

    # Create data reader with static batching
    read_options = DataReadOptions(
        batching=StaticBatching(batch_size=1),  # 1 example per GPU
        example_shuffle_window=10_000,
        batch_shuffle_window=0,
        num_accumulate=1,
        prefetch=4,
        source_encode_mode="prompt",
        target_encode_mode="prompt_response",
        chat_mode=False,
        seed=args.seed,
    )

    gangs = type('Gangs', (), {
        'root': gang,
        'dp': gang,
        'tp': type('Gang', (), {'size': 1})(),
        'sdp': type('Gang', (), {'size': 1})(),
    })()

    data_reader = dataset.create_reader(
        split="train",
        tokenizer=tokenizer,
        gangs=gangs,
        min_seq_len=1,
        max_seq_len=args.max_seq_len,
        options=read_options,
    )

    # Extract batches
    log.info("Starting batch extraction...")
    for batch_idx, batch in enumerate(data_reader):
        if batch_idx >= args.num_batches:
            break

        if gang.rank == 0:
            # Extract batch data
            input_batch, target_batch = batch.as_auto_regressive()
            seqs, seqs_layout = input_batch.as_input()

            # Create attention mask from layout
            if seqs_layout.padded:
                attention_mask = (seqs_layout.position_indices >= 0).to(dtype=torch.long)
            else:
                attention_mask = torch.ones_like(seqs, dtype=torch.long)

            # Create labels from targets and target_mask
            labels = target_batch.seqs.clone()
            if target_batch.target_mask is not None:
                labels = labels.masked_fill(~target_batch.target_mask, -100)

            # Save batch
            batch_data = {
                "input_ids": seqs.cpu(),
                "attention_mask": attention_mask.cpu(),
                "labels": labels.cpu(),
                "batch_idx": batch_idx,
            }

            output_path = args.output_dir / f"batch_{batch_idx:04d}.pt"
            torch.save(batch_data, output_path)

            if batch_idx % 10 == 0:
                log.info(f"Extracted batch {batch_idx}/{args.num_batches}")

    if gang.rank == 0:
        log.info(f"Extraction complete! Saved {args.num_batches} batches to {args.output_dir}")


if __name__ == "__main__":
    main()
