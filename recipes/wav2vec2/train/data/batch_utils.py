# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Batching utilities and strategies for wav2vec2 training.

MIGRATION NOTES:
- Batching logic extracted from fairseq2:e9fbd6/src/fairseq2/datasets/speech.py
- BatchingStrategy replaces v0.4 LengthBatching/StaticBatching classes
- create_bucket_sizes migrated from fairseq2.data
"""

from enum import Enum
from typing import Any, Dict, List

import torch
from torch import Tensor

from fairseq2.data.data_pipeline import DataPipelineBuilder
from fairseq2.datasets import SequenceBatch
from fairseq2.logging import log


class BatchingStrategy(Enum):
    """
    Batching strategies for wav2vec2 training.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/_config.py:16-30
    Classes: StaticBatching, LengthBatching

    NEW IMPLEMENTATION: Combined into enum for cleaner API
    """

    STATIC = "static"  # ORIGINAL: StaticBatching class
    LENGTH = "length"  # ORIGINAL: LengthBatching class


def create_bucket_sizes(
    *,
    max_num_elements: int,
    max_seq_len: int,
    min_seq_len: int = 1,
    num_seqs_multiple_of: int = 1,
) -> list[tuple[int, int]]:
    """
    Create optimal bucket sizes for length-based batching.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/data/__init__.py
    Function: create_bucket_sizes()

    NOTE: Copied exactly from original implementation for numerical parity

    :param max_num_elements:
        The maximum number of elements that each bucket can contain.
    :param max_seq_len:
        The maximum sequence length.
    :param min_seq_len:
        The minimum sequence length.
    :param num_seqs_multiple_of:
        The number of sequences contained in each bucket must be a multiple of this value.
    """
    if max_seq_len > max_num_elements:
        raise ValueError(
            f"`max_seq_len` must be less than or equal to `max_num_elements` ({max_num_elements}), but is {max_seq_len} instead."
        )

    if min_seq_len < 1:
        raise ValueError(
            f"`min_seq_len` must be greater than zero, but is {min_seq_len} instead."
        )

    if min_seq_len > max_seq_len:
        raise ValueError(
            f"`min_seq_len` must be less than or equal to `max_seq_len` ({max_seq_len}), but is {min_seq_len} instead."
        )

    if num_seqs_multiple_of < 1:
        raise ValueError(
            f"`num_seqs_multiple_of` must be greater than or equal to 1, but is {num_seqs_multiple_of} instead."
        )

    if max_num_elements % max_seq_len != 0:
        raise ValueError(
            f"`max_num_elements` must be equal to a multiple of `max_seq_len`, but is {max_num_elements} instead."
        )

    bucket_sizes = []
    seq_len = 1
    bucket_size = max_num_elements

    while seq_len < max_seq_len:
        if seq_len >= min_seq_len:
            bucket_sizes.append((bucket_size, seq_len))

        bucket_size = max_num_elements // (seq_len + 1)
        seq_len = max_num_elements // bucket_size

    bucket_sizes.append((bucket_size, max_seq_len))

    if num_seqs_multiple_of == 1:
        return bucket_sizes

    cropped_bucket_sizes = []
    for bucket_size, seq_len in bucket_sizes:
        if bucket_size > num_seqs_multiple_of:
            bucket_size -= bucket_size % num_seqs_multiple_of
        cropped_bucket_sizes.append((bucket_size, seq_len))

    return cropped_bucket_sizes


# TODO: (cirquit) - this is not needed anymore, but was kind of a partial default for create_bucket_sizes call in length_batching. Using default of 8 in the config for this
# def get_num_seqs_multiple_of(extras: Dict[str, Any]) -> int:
#     """
#     Get num_seqs_multiple_of from options.
#
#     ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:459-465
#     Function: get_num_seqs_multiple_of()
#     """
#     num_seqs_multiple_of = extras.get("num_seqs_multiple_of", 8)
#     assert isinstance(
#         num_seqs_multiple_of, int
#     ), "num_seqs_multiple_of must be an integer"
#     assert num_seqs_multiple_of > 0, "num_seqs_multiple_of must be positive"
#     return num_seqs_multiple_of


class BatchingPipeline:
    """
    Batching pipeline components.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:468-520
    Function: add_bucketing_pipeline()

    NEW IMPLEMENTATION: Split into separate methods for cleaner composition
    """

    @staticmethod
    def add_static_batching(
        builder: DataPipelineBuilder, batch_size: int, drop_remainder: bool = False
    ) -> DataPipelineBuilder:
        """
        Add static batching to pipeline.

        ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:504-509
        In: add_bucketing_pipeline() -> StaticBatching branch
        """
        builder.bucket(batch_size, drop_remainder=drop_remainder)  # ORIGINAL: line 509
        return builder

    @staticmethod
    def add_length_batching(
        builder: DataPipelineBuilder,
        min_audio_len: int,
        max_audio_len: int,
        max_num_elements: int,
        num_seqs_multiple_of: int,
        drop_remainder: bool = False,
    ) -> DataPipelineBuilder:
        """
        Add length-based batching to pipeline.

        ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:478-503
        In: add_bucketing_pipeline() -> LengthBatching branch
        """
        # Bucket by the audio length. - ORIGINAL: line 479
        log.info(f"Using max_num_elements={max_num_elements}!")  # ORIGINAL: line 481

        if max_num_elements % max_audio_len != 0:  # ORIGINAL: line 483
            max_num_elements = (max_num_elements // max_audio_len) * max_audio_len
            log.warning(
                f"`max_num_elements` is rounded to {max_num_elements}"
            )  # ORIGINAL: line 485

        bucket_sizes = create_bucket_sizes(
            min_seq_len=min_audio_len,  # ORIGINAL: line 488
            max_seq_len=max_audio_len,  # ORIGINAL: line 489
            max_num_elements=max_num_elements,  # ORIGINAL: line 490
            num_seqs_multiple_of=num_seqs_multiple_of,
        )

        builder.bucket_by_length(
            bucket_sizes,
            selector="audio_size",  # ORIGINAL: line 498 (was columns parameter) (used to be [*].audio_size)
            min_data_len=min_audio_len,  # ORIGINAL: line 499
            skip_below_min_examples=True,  # ORIGINAL: line 500
            skip_above_max_examples=True,  # ORIGINAL: line 501
            drop_remainder=drop_remainder,  # ORIGINAL: line 502
        )
        return builder

    @staticmethod
    def add_batch_shuffling(
        builder: DataPipelineBuilder, batch_shuffle_window: int, seed: int
    ) -> DataPipelineBuilder:
        """
        Add batch shuffling to pipeline.

        ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:513-520
        In: add_bucketing_pipeline() -> batch shuffling logic
        """
        # Shuffle buckets. - ORIGINAL: line 513
        # assert (
        #    batch_shuffle_window > 0  # ORIGINAL: line 514-516
        # ), f"{batch_shuffle_window=}: can apply full batch shuffling which may result in OOM"
        if batch_shuffle_window == 0:
            log.warning(
                f"Applying full batch shuffling ({batch_shuffle_window=}) may result in OOM."
            )
        elif batch_shuffle_window > 0:
            log.info(f"Shuffling inside batch window {batch_shuffle_window=}.")

        if batch_shuffle_window != 1:  # ORIGINAL: line 518
            builder.shuffle(batch_shuffle_window, seed)  # ORIGINAL: line 519
        return builder


def create_sequence_batch(
    batch_dict: Dict[str, Any], no_padding: bool = True
) -> SequenceBatch:
    """
    Convert batch dictionary to SequenceBatch with proper sequence length handling.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:296-309
    Function: to_batch()

    v0.5 Migration: Replaced PaddingMask usage with direct SequenceBatch construction
    using seq_lens from Collater output.
    """
    audio_feature = batch_dict["audio_feature"]  # ORIGINAL: line 302

    if no_padding:  # ORIGINAL: line 303
        # no_padding=True: All sequences cropped to same length, no padding needed
        # audio_feature is a plain Tensor from Collater(pad_value=None)
        return SequenceBatch(audio_feature, seq_lens=None, example=batch_dict)
    else:
        # no_padding=False: Sequences are padded, need actual sequence lengths
        # audio_feature is SequenceData dict from Collater(pad_value=0)
        if isinstance(audio_feature, dict) and "seq_lens" in audio_feature:
            seqs = audio_feature["seqs"]
            seq_lens = audio_feature["seq_lens"]

            # Convert seq_lens to list[int] if it's a tensor
            if hasattr(seq_lens, "tolist"):
                seq_lens = seq_lens.tolist()
            else:
                seq_lens = list(seq_lens)

            return SequenceBatch(seqs, seq_lens=seq_lens, example=batch_dict)
        else:
            # Fallback: assume uniform lengths (should not happen with proper Collater setup)
            return SequenceBatch(audio_feature, seq_lens=None, example=batch_dict)
