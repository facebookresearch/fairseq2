# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.functional import layer_norm

from fairseq2.data.audio import AudioDecoder
from fairseq2.data.data_pipeline import (
    CollateOptionsOverride,
    Collater,
    DataPipelineBuilder,
    FileMapper,
)
from fairseq2.data.tokenizers import TokenEncoder


def add_audio_file_loading(
    builder: DataPipelineBuilder, audio_dir: Path, cached_fd_count: int, selector: str
) -> DataPipelineBuilder:
    """
    Load audio files from disk into memory via ``FileMapper``.

    Transforms relative audio paths into absolute paths and memory-maps the files.

    :param audio_dir:
        Root directory for resolving relative audio paths.
    :param cached_fd_count:
        Number of files to cache in an LRU cache. ``FileMapper`` will keep the last
        ``cached_fd_count`` files memory-mapped, which is especially useful when
        reading multiple slices from the same audio file.
    :param selector:
        JSONPath selector for the field containing audio paths (e.g., ``"[*].audio"``).
    """

    file_mapper = FileMapper(audio_dir, cached_fd_count=cached_fd_count)
    return builder.map(file_mapper, selector=selector)


def add_audio_decoding(
    builder: DataPipelineBuilder,
    dtype: torch.dtype,
    normalize_audio: bool,
    npc: int,
    selector: str,
) -> DataPipelineBuilder:
    """Add audio decoding to pipeline by creating a ``fairseq2.data._memory.MemoryBlock`` at
    the selector. Waveforms are in ``torch.float32`` if ``normalize_audio``, else ``dtype``.
    """

    audio_decoder = AudioDecoder(dtype=torch.float32 if normalize_audio else dtype)
    return builder.map(
        audio_decoder,
        selector=selector,
        num_parallel_calls=npc,
    )


def add_layernorm(
    builder: DataPipelineBuilder, dtype: torch.dtype, selector: str
) -> DataPipelineBuilder:
    """Applies ``torch.nn.functional.layer_norm`` onto the waveforms and casts them to ``dtype``."""

    def normalize(waveform: Tensor) -> Tensor:
        with torch.no_grad():
            waveform = layer_norm(waveform, waveform.shape)

        return waveform.to(dtype)

    return builder.map(normalize, selector=selector)


def encode_text(
    builder: DataPipelineBuilder, token_encoder: TokenEncoder, npc: int, selector: str
) -> DataPipelineBuilder:
    """Encodes text with ``token_encoder`` number of parallel calls ``npc``."""

    return builder.map(token_encoder, selector=selector, num_parallel_calls=npc)


def collate_with_pad_ix(
    builder: DataPipelineBuilder, pad_idx: int, no_padding: bool, selector: str
) -> DataPipelineBuilder:
    """Collates text tensors at ``selector`` with ``pad_idx`` and audio with:
    - if ``no_padding``: ``None``
    - otherwise: ``0``

    The remaining keys (str)  or (int) are collated as lists.
    """

    text_collate_opts = CollateOptionsOverride(selector, pad_value=pad_idx)  # type: ignore[arg-type]
    collater = Collater(
        pad_value=None if no_padding else 0, overrides=[text_collate_opts]
    )
    return builder.map(collater)


def filter_by_min_max_audio_size(
    builder: DataPipelineBuilder,
    min_audio_len: int,
    max_audio_len: int,
    audio_size_selector: str,
) -> DataPipelineBuilder:
    """Expects the data to have an integer `audio_size_selector` field."""

    def is_right_length(example: dict[str, Any]) -> bool:
        size: int = example[audio_size_selector]
        return min_audio_len <= size <= max_audio_len

    return builder.filter(is_right_length)
