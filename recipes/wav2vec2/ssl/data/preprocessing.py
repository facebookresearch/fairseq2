# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from fairseq2.data.audio import AudioDecoder
from fairseq2.data.data_pipeline import DataPipelineBuilder, FileMapper


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


class AudioCropper:
    """Crops audio sequences to maximum length.
    Emulates the JSONPath access scheme naively to be independent of hardcoded selector keys.
    """

    def __init__(
        self,
        max_audio_len: int,
        seed: int,
        crop_to_batch_minimal_size: bool,
        audio_feature_selector: str,
    ) -> None:
        self.rng: np.random.RandomState = np.random.RandomState(seed)
        self.max_audio_len: int = max_audio_len
        self.crop_to_batch_minimal_size: bool = crop_to_batch_minimal_size
        self.audio_feature_selector = audio_feature_selector

    def crop_audios_in_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Crop audio sequences in a batch."""
        if self.crop_to_batch_minimal_size:
            min_audio_len_batch = min(
                (
                    AudioCropper._get_nested(item, self.audio_feature_selector).size(0)  # type: ignore
                    for item in batch
                )
            )
            crop_size = min(self.max_audio_len, min_audio_len_batch)
        else:
            crop_size = self.max_audio_len

        for item in batch:

            audio = AudioCropper._get_nested(item, self.audio_feature_selector)
            audio_size = audio.size(0)  # type: ignore
            if audio_size > crop_size:
                start = self.rng.randint(0, audio_size - crop_size)
                value = audio[start : start + crop_size]  # type: ignore
                AudioCropper._set_nested(item, self.audio_feature_selector, value)
        return batch

    @staticmethod
    def _get_nested(d: dict[str, Any], selector: str) -> dict[str, Any]:
        """Getter that emulates the JSONPath selector from DataPipeline in Python."""
        keys = selector.split(".")
        for key in keys:
            d = d[key]
        return d

    @staticmethod
    def _set_nested(d: dict[str, Any], selector: str, value: Any) -> None:
        """Setter that emulates the JSONPath selector from DataPipeline in Python."""
        keys = selector.split(".")
        # navigate to parent of last key
        for key in keys[:-1]:
            d = d[key]
        # set last key
        last_key = keys[-1]
        d[last_key] = value


def add_audio_cropping(
    builder: DataPipelineBuilder,
    seed: int,
    max_audio_len: int,
    crop_to_batch_minimal_size: bool,
    audio_feature_selector: str,
) -> DataPipelineBuilder:
    """Crop long audios to `max_audio_len`."""
    audio_cropper = AudioCropper(
        max_audio_len,
        seed=seed,
        crop_to_batch_minimal_size=crop_to_batch_minimal_size,
        audio_feature_selector=audio_feature_selector,
    )
    return builder.map(audio_cropper.crop_audios_in_batch)
