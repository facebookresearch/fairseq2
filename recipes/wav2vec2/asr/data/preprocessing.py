# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.nn.functional import layer_norm

from fairseq2.data.audio import AudioDecoder
from fairseq2.data.data_pipeline import (
    CollateOptionsOverride,
    Collater,
    DataPipelineBuilder,
    FileMapper,
)
from fairseq2.data.tokenizers import TokenEncoder, Tokenizer
from fairseq2.logging import log

try:
    import torchaudio  # type: ignore
except ImportError:
    torchaudio = None
    log.warning(
        "torchaudio is not installed. Please install it with `pip install torchaudio`."
    )


@torch.no_grad()
def apply_audio_normalization(waveform: Tensor) -> Tensor:
    """Normalize audio to zero mean and unit variance."""
    return layer_norm(waveform, waveform.shape)


@torch.no_grad()
def convert_to_mono(waveform: Tensor) -> Tensor:
    """Convert multi-channel audio to mono by averaging channels."""
    if waveform.dim() == 2:
        # reduce channels inplace to save the memory
        size = waveform.size(1)
        result = reduce(
            torch.Tensor.add_, [waveform[:, i] for i in range(1, size)], waveform[:, 0]
        )
        waveform = result
        waveform /= size

    return waveform


@torch.no_grad()
def apply_freq_mask(spec: Tensor, freq_mask_param: int = 80) -> Tensor:
    """Apply frequency masking to the spectrogram."""
    n_freq = spec.size(-2)

    assert freq_mask_param < n_freq
    fmask_len = random.randint(20, freq_mask_param)
    fmask_i = random.randint(0, (n_freq - fmask_len - 1))

    masked_spec = spec.clone()
    masked_spec[:, fmask_i : (fmask_i + fmask_len)] = 0.0
    return masked_spec


@torch.no_grad()
def apply_time_mask(spec: Tensor, time_mask_param: int = 80) -> Tensor:
    """Apply time masking to the spectrogram."""
    n_t = spec.size(-1)

    time_mask_param = min(120, int(n_t / 4))
    assert time_mask_param < n_t
    tmask_len = random.randint(0, time_mask_param)
    tmask_i = random.randint(0, (n_t - tmask_len - 1))

    masked_spec = spec.clone()
    masked_spec[..., tmask_i : (tmask_i + tmask_len)] = 0.0
    return masked_spec


@torch.no_grad()
def apply_spec_augment(
    waveform: Tensor,
    n_fft: int = 400,
    win_len: Optional[int] = None,
    hop_len: Optional[int] = None,
    power: int | None = None,
    freq_mask_param: int = 80,
    time_mask_param: int = 80,
) -> Tensor:
    """Apply SpecAugment with frequency and time masking."""
    # Get spectrogram
    spectrogram = torchaudio.transforms.Spectrogram(  # type: ignore
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )(waveform)

    # Augment
    spectrogram_aug = apply_freq_mask(spectrogram, freq_mask_param)
    spectrogram_aug = apply_time_mask(spectrogram_aug, time_mask_param)

    # Convert back to waveform
    inverse_spec = torchaudio.transforms.InverseSpectrogram()  # type: ignore
    waveform_aug: Tensor = inverse_spec(spectrogram_aug)
    return waveform_aug


@torch.no_grad()
def postprocess_waveform(
    waveform: Tensor,
    normalize_audio: bool,
    dtype: torch.dtype,
    spec_aug_p: Optional[float] = None,
    spec_aug_freq_mask_param: int = 80,
    spec_aug_time_mask_param: int = 80,
) -> Tensor:
    """Post-process audio waveform with normalization and optional SpecAugment."""
    # Handle multi-channel audio
    waveform = convert_to_mono(waveform)

    # Apply normalization
    if normalize_audio:
        waveform = apply_audio_normalization(waveform)

    # Apply SpecAugment
    if spec_aug_p is not None and random.random() < spec_aug_p:
        waveform = apply_spec_augment(
            waveform,
            freq_mask_param=spec_aug_freq_mask_param,
            time_mask_param=spec_aug_time_mask_param,
        )

    return waveform.to(dtype)


class AudioCropper:
    """Crops audio sequences to maximum length."""

    audio_feature: str = "audio_feature"

    def __init__(
        self, max_audio_len: int, seed: int, crop_to_batch_minimal_size: bool = False
    ) -> None:
        self.rng: np.random.RandomState = np.random.RandomState(seed)
        self.max_audio_len: int = max_audio_len
        self.crop_to_batch_minimal_size: bool = crop_to_batch_minimal_size

    def crop_audios_in_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Crop audio sequences in a batch by ``self.max_audio_len`` or
        by ``min(item.size(0) for item in batch)`` if ``self.crop_to_batch_minimal_size = True``.
        """
        if self.crop_to_batch_minimal_size:
            min_audio_len_batch = min(
                (item[self.audio_feature].size(0) for item in batch)
            )
            crop_size = min(self.max_audio_len, min_audio_len_batch)
        else:
            crop_size = self.max_audio_len

        for item in batch:
            audio = item[self.audio_feature]
            if audio.size(0) > crop_size:
                start = self.rng.randint(0, audio.size(0) - crop_size)
                item[self.audio_feature] = audio[start : start + crop_size]

        return batch


class AudioProcessingPipeline:
    """Composable audio processing pipeline builder."""

    @staticmethod
    def add_path_resolution(
        builder: DataPipelineBuilder,
        audio_dir: Optional[Path],
        cached_fd_count: int,
    ) -> DataPipelineBuilder:
        """
        Add audio path resolution to pipeline via ``FileMapper``.

        :param audio_dir:
            Optional prefix for the audio directory
        :param cached_fd_count:
            Enables an LRU cache on the last ``cached_fd_count`` files read.
            ``FileMapper`` will memory map all the cached file,
            so this is especially useful for reading several slices of the same file.
        """

        selector = "[*].audio"
        file_mapper = FileMapper(audio_dir, cached_fd_count=cached_fd_count)
        return builder.map(file_mapper, selector=selector)

    @staticmethod
    def add_audio_decoding(
        builder: DataPipelineBuilder,
        dtype: torch.dtype,
        normalize_audio: bool,
        npc: int = 10,
    ) -> DataPipelineBuilder:
        """Add audio decoding to pipeline by creating a ``fairseq2.data._memory.MemoryBlock`` at
        the selector. Waveforms are in ``torch.float32`` if ``normalize_audio``, else ``dtype``.
        """

        audio_decoder = AudioDecoder(dtype=torch.float32 if normalize_audio else dtype)
        return builder.map(
            audio_decoder,
            selector="[*].audio.data",
            num_parallel_calls=npc,
        )

    @staticmethod
    def add_layernorm(
        builder: DataPipelineBuilder, dtype: torch.dtype
    ) -> DataPipelineBuilder:
        """Applies ``torch.nn.functional.layer_norm`` onto the waveforms and casts them to ``dtype``."""

        def normalize(waveform: Tensor) -> Tensor:
            with torch.no_grad():
                waveform = layer_norm(waveform, waveform.shape)

            return waveform.to(dtype)

        return builder.map(normalize, selector="[*].audio.data.waveform")


class TextProcessingPipeline:
    """Composable text processing pipeline builder."""

    _text_encoder: TokenEncoder
    _pad_idx: int | None

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._text_encoder = tokenizer.create_encoder()
        self._pad_idx = tokenizer.vocab_info.pad_idx

    def encode_text(
        self, builder: DataPipelineBuilder, npc: int
    ) -> DataPipelineBuilder:
        """Encodes text with ``self._text_encoder`` number of parallel calls ``npc``."""

        return builder.map(
            self._text_encoder, selector="[*].text", num_parallel_calls=npc
        )

    def collate_with_pad_ix(
        self, builder: DataPipelineBuilder, no_padding: bool
    ) -> DataPipelineBuilder:
        """Collates text tensors with ``self._pad_idx`` and audio with:
        - if ``no_padding``: ``None``
        - otherwise: ``0``

        The remaining keys (str)  or (int) are collated as lists.
        """

        text_collate_opts = CollateOptionsOverride("text", pad_value=self._pad_idx)
        collater = Collater(
            pad_value=None if no_padding else 0, overrides=[text_collate_opts]
        )
        return builder.map(collater)
