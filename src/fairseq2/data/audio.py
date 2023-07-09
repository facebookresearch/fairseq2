# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Optional, TypedDict

from torch import Tensor

from fairseq2 import _DOC_MODE
from fairseq2.memory import MemoryBlock
from fairseq2.typing import DataType, Device

if TYPE_CHECKING or _DOC_MODE:

    class AudioDecoder:
        def __init__(
            self,
            dtype: Optional[DataType] = None,
            device: Optional[Device] = None,
            pin_memory: bool = False,
        ) -> None:
            ...

        def __call__(self, memory_block: MemoryBlock) -> "AudioDecoderOutput":
            ...

else:
    from fairseq2.C.data.audio import AudioDecoder as AudioDecoder

    AudioDecoder.__module__ = __name__


class AudioDecoderOutput(TypedDict):
    audio: Tensor
    sample_rate: int
    format: int
