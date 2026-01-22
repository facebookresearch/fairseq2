# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, final

from fairseq2n import DOC_MODE
from torch import Tensor

from fairseq2.data._memory import MemoryBlock
from fairseq2.device import Device

if TYPE_CHECKING or DOC_MODE:

    @final
    class ImageDecoder:
        def __init__(
            self,
            device: Device | None = None,
            pin_memory: bool = False,
        ) -> None: ...

        def __call__(self, memory_block: MemoryBlock) -> ImageDecoderOutput: ...

else:
    from fairseq2n.bindings.data.image import ImageDecoder as ImageDecoder  # noqa: F401


class ImageDecoderOutput(TypedDict):
    bit_depth: float
    color_type: float
    channels: float
    height: float
    width: float
    image: Tensor
