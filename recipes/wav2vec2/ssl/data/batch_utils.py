# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.data.data_pipeline import DataPipelineBuilder, create_bucket_sizes
from fairseq2.logging import log


def add_length_batching(
    builder: DataPipelineBuilder,
    min_audio_len: int,
    max_audio_len: int,
    max_num_elements: int,
    num_seqs_multiple_of: int,
    drop_remainder: bool,
    selector: str,
) -> DataPipelineBuilder:
    """Add length-based batching to pipeline."""
    log.info(f"Using max_num_elements={max_num_elements}!")

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
        selector=selector,
        min_data_len=min_audio_len,
        skip_below_min_examples=True,
        skip_above_max_examples=True,
        drop_remainder=drop_remainder,
    )


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
