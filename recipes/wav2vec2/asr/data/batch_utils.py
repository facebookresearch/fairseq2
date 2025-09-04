# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, cast

from fairseq2.data.data_pipeline import DataPipelineBuilder, create_bucket_sizes
from fairseq2.datasets import SequenceBatch
from fairseq2.logging import log


class BatchingStrategy(Enum):
    """Batching strategies for wav2vec2 training."""

    STATIC = "STATIC"
    LENGTH = "LENGTH"


class BatchingPipeline:
    """Batching pipeline components."""

    @staticmethod
    def filter_by_min_max_audio_length(
        builder: DataPipelineBuilder, min_audio_length: int, max_audio_length: int
    ) -> DataPipelineBuilder:
        """Filters samples by ``min_audio_length`` and ``max_audio_length``."""

        def skip(example: dict[str, object]) -> bool:
            audio_length = cast(int, example["audio_size"])

            return audio_length >= min_audio_length and audio_length <= max_audio_length

        return builder.filter(skip)

    @staticmethod
    def add_static_batching(
        builder: DataPipelineBuilder, batch_size: int, drop_remainder: bool
    ) -> DataPipelineBuilder:
        """Add static batching to pipeline."""
        return builder.bucket(batch_size, drop_remainder=drop_remainder)

    @staticmethod
    def add_length_batching(
        builder: DataPipelineBuilder,
        min_audio_len: int,
        max_audio_len: int,
        max_num_elements: int,
        num_seqs_multiple_of: int,
        drop_remainder: bool,
    ) -> DataPipelineBuilder:
        """Add length-based batching to pipeline."""
        # Bucket by the audio length.
        log.info(f"Using length batching with max_num_elements={max_num_elements}!")

        if max_num_elements % max_audio_len != 0:
            max_num_elements = (max_num_elements // max_audio_len) * max_audio_len
            log.warning(f"`max_num_elements` is rounded to {max_num_elements}")

        bucket_sizes = create_bucket_sizes(
            min_seq_len=min_audio_len,
            max_seq_len=max_audio_len,
            max_num_elements=max_num_elements,
            num_seqs_multiple_of=num_seqs_multiple_of,
        )

        return builder.bucket_by_length(
            bucket_sizes,
            selector="audio_size",
            min_data_len=min_audio_len,
            skip_below_min_examples=True,
            skip_above_max_examples=True,
            drop_remainder=drop_remainder,
        )

    @staticmethod
    def add_batch_shuffling(
        builder: DataPipelineBuilder, batch_shuffle_window: int, seed: int
    ) -> DataPipelineBuilder:
        """Add batch shuffling to pipeline."""
        if batch_shuffle_window == 0:
            log.warning(
                f"Applying full batch shuffling ({batch_shuffle_window=}) may result in OOM."
            )
        elif batch_shuffle_window > 0:
            log.info(f"Shuffling inside batch window {batch_shuffle_window=}.")

        if batch_shuffle_window != 1:
            builder.shuffle(batch_shuffle_window, seed)
        return builder


def create_sequence_batch(
    batch_dict: Dict[str, Any], no_padding: bool = True
) -> SequenceBatch:
    """
    Convert batch dictionary to SequenceBatch with proper sequence length handling.
    """
    audio_feature = batch_dict["audio_feature"]

    if no_padding:
        # no_padding=True: All sequences cropped to same length, no padding needed
        # audio_feature is a plain Tensor from Collater(pad_value=None)
        return SequenceBatch(audio_feature, seq_lens=None, example=batch_dict)
    else:
        # no_padding=False: Sequences are padded, need actual sequence lengths
        # audio_feature is SequenceData dict from Collater(pad_value=0)
        if isinstance(audio_feature, dict) and "seq_lens" in audio_feature:
            seqs = audio_feature["seqs"]
            seq_lens = audio_feature["seq_lens"]
            return SequenceBatch(seqs, seq_lens=seq_lens, example=batch_dict)
        else:
            # Fallback: assume uniform lengths (should not happen with proper Collater setup)
            return SequenceBatch(
                audio_feature, seq_lens=None, example=batch_dict  # type: ignore
            )
