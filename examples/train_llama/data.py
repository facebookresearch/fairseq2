"""Data pipeline creation for training."""

import random
from typing import Any

import torch
from torch import Tensor

from fairseq2.data.data_pipeline import DataPipeline, read_sequence
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets import SequenceBatch
from fairseq2.device import Device
from fairseq2.gang import Gangs
from fairseq2.logging import log

from config import TrainingConfig


def create_train_eval_pipelines(
    dataset_name: str,
    tokenizer: Tokenizer,
    gangs: Gangs,
    config: TrainingConfig,
) -> tuple[DataPipeline, DataPipeline]:
    """
    Creates train and eval data pipelines with 90/10 split.

    The pipeline:
    1. Reads text data and shuffles it
    2. Splits into train (90%) and eval (10%)
    3. Tokenizes and batches each split
    4. Train split is re-shuffled, eval is not

    Args:
        dataset_name: Name of the dataset asset to load
        tokenizer: Tokenizer for encoding text
        gangs: Gang abstraction for distributed training
        config: Training configuration

    Returns:
        Tuple of (train_pipeline, eval_pipeline)
    """
    log.info("Creating train/eval data pipelines for dataset '{}'", dataset_name)

    # Create a token encoder from the tokenizer
    text_encoder = tokenizer.create_encoder(mode="default", device=gangs.device)

    # Synthetic text dataset
    sample_instructions = [
        "Translate the following English text to French: Hello, how are you?",
        "Write a Python function to calculate the Fibonacci sequence.",
        "Explain the concept of machine learning in simple terms.",
        "What is the capital of France?",
        "Describe the process of photosynthesis.",
    ] * 200  # 1000 total examples

    # Calculate split sizes
    total_size = len(sample_instructions)
    eval_size = int(total_size * config.eval_split_ratio)
    train_size = total_size - eval_size

    log.info("Data split: {} train, {} eval ({:.1f}% eval)",
             train_size, eval_size, config.eval_split_ratio * 100)

    # Shuffle all data first, then split
    rng = random.Random(config.seed)
    shuffled_instructions = sample_instructions.copy()
    rng.shuffle(shuffled_instructions)

    train_data = shuffled_instructions[:train_size]
    eval_data = shuffled_instructions[train_size:]

    # Create train pipeline
    train_pipeline = (
        read_sequence(train_data)
        .shard(gangs.dp.rank, gangs.dp.size, allow_uneven=True)
        .shuffle(shuffle_window=1000, seed=config.seed)
        .map(text_encoder)
        .map(lambda x: _prepare_sequence(x, config.max_seq_len, tokenizer.vocab_info.pad_idx))
        .bucket(bucket_size=config.batch_size)
        .map(lambda batch: _collate_batch(batch, gangs.device))
        .prefetch(num_examples=4)
        .and_return()
    )

    # Create eval pipeline (no shuffling for deterministic results)
    eval_pipeline = (
        read_sequence(eval_data)
        .shard(gangs.dp.rank, gangs.dp.size, allow_uneven=True)
        .map(text_encoder)
        .map(lambda x: _prepare_sequence(x, config.max_seq_len, tokenizer.vocab_info.pad_idx))
        .bucket(bucket_size=config.batch_size)
        .map(lambda batch: _collate_batch(batch, gangs.device))
        .prefetch(num_examples=4)
        .and_return()
    )

    return train_pipeline, eval_pipeline


def _prepare_sequence(encoded_output: Any, max_len: int, pad_idx: int) -> dict[str, Tensor]:
    """
    Prepares an encoded sequence for causal language modeling.

    Truncates or pads the sequence to the specified maximum length.

    Args:
        encoded_output: Output from tokenizer encoder (dict or tensor)
        max_len: Maximum sequence length
        pad_idx: Padding token index

    Returns:
        Dictionary with 'seqs' key containing processed sequence tensor [max_len]
    """
    # Handle both dict and direct tensor output from encoder
    if isinstance(encoded_output, dict):
        seq = encoded_output["seqs"]
    else:
        seq = encoded_output

    if seq.size(0) > max_len:
        # Truncate to max length
        processed_seq = seq[:max_len]
    elif seq.size(0) < max_len:
        # Pad to max length
        padding = torch.full(
            (max_len - seq.size(0),),
            pad_idx,
            dtype=seq.dtype,
            device=seq.device,
        )
        processed_seq = torch.cat([seq, padding])
    else:
        processed_seq = seq

    return {"seqs": processed_seq}


def _collate_batch(batch: list[dict[str, Any]], device: Device) -> SequenceBatch:
    """
    Collates a list of examples into a SequenceBatch.

    Args:
        batch: List of dictionaries, each containing 'seqs' key
        device: Target device for the batch

    Returns:
        SequenceBatch ready for training
    """
    # Extract sequences from each example in the batch and stack them
    seqs_list = [example["seqs"] for example in batch]
    seqs = torch.stack(seqs_list)

    # Create sequence lengths as a list
    seq_lens = [seqs.size(1) for _ in range(len(batch))]

    return SequenceBatch(
        seqs=seqs,
        seq_lens=seq_lens,
    )
