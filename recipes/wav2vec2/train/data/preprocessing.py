# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Audio preprocessing pipeline components for wav2vec2 training.
All functions are pure and composable.

MIGRATION NOTES:
- Migrated from fairseq2:e9fbd6/src/fairseq2/datasets/speech.py
- All audio processing logic extracted for clean separation
- Maintains 1:1 numerical parity with original implementation
"""

from __future__ import annotations

import random
from functools import partial, reduce
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.nn.functional import layer_norm

from fairseq2.data import audio
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.data.data_pipeline import DataPipelineBuilder, FileMapper
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
    """
    Normalize audio to zero mean and unit variance.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:137
    Function: postprocess() -> layer_norm(waveform, waveform.shape)
    """
    return layer_norm(waveform, waveform.shape)


@torch.no_grad()
def convert_to_mono(waveform: Tensor) -> Tensor:
    """
    Convert multi-channel audio to mono by averaging channels.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:127-134
    Function: postprocess() -> channel reduction logic
    """
    if waveform.dim() == 2:
        # reduce channels inplace to save the memory
        size = waveform.size(1)
        # result = torch.sum(waveform, dim=1) / size  # Original used reduce() but this is equivalent
        # waveform = result
        result = reduce(
            torch.Tensor.add_, [waveform[:, i] for i in range(1, size)], waveform[:, 0]
        )
        waveform = result
        waveform /= size

    return waveform


@torch.no_grad()
def apply_freq_mask(spec: Tensor, freq_mask_param: int = 80) -> Tensor:
    """
    Apply frequency masking to the spectrogram.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:54-64
    Function: freq_mask()
    """
    n_freq = spec.size(-2)

    assert freq_mask_param < n_freq
    fmask_len = random.randint(20, freq_mask_param)
    fmask_i = random.randint(0, (n_freq - fmask_len - 1))

    masked_spec = spec.clone()
    masked_spec[:, fmask_i : (fmask_i + fmask_len)] = 0.0
    return masked_spec


@torch.no_grad()
def apply_time_mask(spec: Tensor, time_mask_param: int = 80) -> Tensor:
    """
    Apply time masking to the spectrogram.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:67-78
    Function: time_mask()
    """
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
    """
    Apply SpecAugment with frequency and time masking.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:82-116
    Function: _apply_spec_augment()
    """
    # get spectrogram - ORIGINAL: line 98-103
    spectrogram = torchaudio.transforms.Spectrogram(  # type: ignore
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )(waveform)

    # augment
    spectrogram_aug = apply_freq_mask(spectrogram, freq_mask_param)
    spectrogram_aug = apply_time_mask(spectrogram_aug, time_mask_param)

    # convert back to waveform - ORIGINAL: line 108-116
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
    """
    Post-process audio waveform with normalization and optional SpecAugment.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:119-146
    Function: postprocess()
    """
    # Handle multi-channel audio - ORIGINAL: line 127-134
    waveform = convert_to_mono(waveform)

    # Apply normalization - ORIGINAL: line 136-137
    if normalize_audio:
        waveform = apply_audio_normalization(waveform)

    # Apply SpecAugment - ORIGINAL: line 139-145
    if spec_aug_p is not None and random.random() < spec_aug_p:
        waveform = apply_spec_augment(
            waveform,
            freq_mask_param=spec_aug_freq_mask_param,
            time_mask_param=spec_aug_time_mask_param,
        )

    return waveform.to(dtype)  # ORIGINAL: line 146


class AudioCropper:
    """
    Crops audio sequences to maximum length.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:149-178
    Class: AudioCropper
    """

    audio_feature: str = "audio_feature"  # ORIGINAL: line 150

    def __init__(
        self, max_audio_len: int, seed: int, crop_to_batch_minimal_size: bool = False
    ) -> None:
        self.rng: np.random.RandomState = np.random.RandomState(seed)
        self.max_audio_len: int = max_audio_len
        self.crop_to_batch_minimal_size: bool = crop_to_batch_minimal_size

    def crop_audios_in_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Crop audio sequences in a batch.

        ORIGINAL: line 159-178
        Method: crop_audios_in_batch()
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
    """
    Composable audio processing pipeline builder.

    NEW IMPLEMENTATION: Created for v0.5 API compatibility
    Replaces original static methods from GenericSpeechDataset
    """

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

        ORIGINAL EQUIVALENT: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:328-336
        In: add_audio_decoding() -> FileMapper logic
        """
        selector = "[*].audio"
        file_mapper = FileMapper(audio_dir, cached_fd_count=cached_fd_count)
        builder.map(file_mapper, selector=selector)
        return builder

    @staticmethod
    def add_audio_decoding(
        builder: DataPipelineBuilder,
        dtype: torch.dtype,
        normalize_audio: bool,
        npc: int = 10,
    ) -> DataPipelineBuilder:
        """
        Add audio decoding to pipeline.

        ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:338-344
        In: add_audio_decoding() -> AudioDecoder setup and mapping
        """
        # Decode audio. - ORIGINAL: line 338-344
        audio_decoder = AudioDecoder(
            dtype=torch.float32 if normalize_audio else dtype  # ORIGINAL: line 340
        )
        builder.map(
            audio_decoder,
            selector="[*].audio.data",
            num_parallel_calls=npc,  # ORIGINAL: line 341-344
        )
        return builder

    @staticmethod
    def add_waveform_processing(
        builder: DataPipelineBuilder,
        normalize_audio: bool,
        dtype: torch.dtype,
        spec_aug_p: Optional[float] = None,
        spec_aug_freq_mask_param: int = 80,
        spec_aug_time_mask_param: int = 80,
    ) -> DataPipelineBuilder:
        """
        Add waveform processing (normalization + SpecAugment).

        ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:348-380
        In: audio_post_process() -> else branch for waveform processing
        """
        builder.map(
            partial(
                postprocess_waveform,
                normalize_audio=normalize_audio,
                dtype=dtype,
                spec_aug_p=spec_aug_p,
                spec_aug_freq_mask_param=spec_aug_freq_mask_param,
                spec_aug_time_mask_param=spec_aug_time_mask_param,
            ),
            selector="[*].audio.data.waveform",  # ORIGINAL: line 375 selector
        )
        return builder

    @staticmethod
    def add_fbank_processing(
        builder: DataPipelineBuilder, dtype: torch.dtype, npc: int = 10
    ) -> DataPipelineBuilder:
        """
        Add filterbank feature extraction.

        ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:109-123
        In: audio_post_process() -> WaveformToFbankConverter setup
        """
        fbank_converter = WaveformToFbankConverter(
            num_mel_bins=80,  # ORIGINAL: line 112
            waveform_scale=2**15,  # ORIGINAL: line 113
            channel_last=True,  # ORIGINAL: line 114
            standardize=True,  # ORIGINAL: line 115
            dtype=dtype,  # ORIGINAL: line 116
        )

        builder.map(
            fbank_converter,
            selector="[*].audio.data",  # ORIGINAL: line 360 selector
            num_parallel_calls=npc,  # ORIGINAL: line 361
        )
        return builder

    @staticmethod
    def add_feature_renaming(
        builder: DataPipelineBuilder, use_fbank: bool
    ) -> DataPipelineBuilder:
        """
        Add feature renaming step.

        ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:312-318
        Function: rename_feature()
        """

        def rename_feature(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            # ORIGINAL: line 312-318 - operates on batch, not individual examples
            for example in batch:
                if use_fbank and "fbank" in example["audio"]["data"]:
                    example["audio_feature"] = example["audio"]["data"].pop("fbank")
                elif not use_fbank and "waveform" in example["audio"]["data"]:
                    example["audio_feature"] = example["audio"]["data"].pop("waveform")
            return batch

        builder.map(rename_feature)  # NO SELECTOR - operates on full batch

        return builder

    @staticmethod
    def add_audio_cropping(
        builder: DataPipelineBuilder,
        seed: int,
        max_audio_len: int,
        crop_to_batch_minimal_size: bool,
    ) -> DataPipelineBuilder:
        # Crop long audios to `max_audio_len`.
        audio_cropper = AudioCropper(
            max_audio_len,
            seed=seed,
            crop_to_batch_minimal_size=crop_to_batch_minimal_size,
        )
        builder.map(audio_cropper.crop_audios_in_batch)
        return builder
