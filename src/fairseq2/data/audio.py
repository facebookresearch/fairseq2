# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypedDict, Union

from fairseq2n import DOC_MODE
from torch import Tensor
from typing_extensions import NotRequired

from fairseq2.memory import MemoryBlock
from fairseq2.typing import DataType, Device

if TYPE_CHECKING or DOC_MODE:

    class AudioDecoder:
        def __init__(
            self,
            dtype: Optional[DataType] = None,
            device: Optional[Device] = None,
            pin_memory: bool = False,
        ) -> None:
            ...

        def __call__(self, memory_block: MemoryBlock) -> AudioDecoderOutput:
            ...

    class WaveformToFbankConverter:
        def __init__(
            self,
            num_mel_bins: int = 80,
            waveform_scale: float = 1.0,
            channel_last: bool = False,
            standardize: bool = False,
            keep_waveform: bool = False,
            dtype: Optional[DataType] = None,
            device: Optional[Device] = None,
            pin_memory: bool = False,
        ) -> None:
            ...

        def __call__(self, waveform: WaveformToFbankInput) -> WaveformToFbankOutput:
            ...

else:
    from fairseq2n.bindings.data.audio import AudioDecoder as AudioDecoder
    from fairseq2n.bindings.data.audio import (
        WaveformToFbankConverter as WaveformToFbankConverter,
    )

    def _set_module_name() -> None:
        for t in [AudioDecoder, WaveformToFbankConverter]:
            t.__module__ = __name__

    _set_module_name()


class AudioDecoderOutput(TypedDict):
    waveform: Tensor
    sample_rate: float
    format: int


class WaveformToFbankInput(TypedDict):
    waveform: Tensor
    sample_rate: Union[int, float]


class WaveformToFbankOutput(TypedDict):
    fbank: Tensor
    waveform: NotRequired[Tensor]
    sample_rate: float
