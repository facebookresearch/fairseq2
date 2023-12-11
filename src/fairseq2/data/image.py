# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Optional, TypedDict

from fairseq2n import DOC_MODE
from torch import Tensor

from fairseq2.memory import MemoryBlock
from fairseq2.typing import Device

if TYPE_CHECKING or DOC_MODE:

    class ImageDecoder:
        def __init__(
            self,
            device: Optional[Device] = None,
            pin_memory: bool = False,
        ) -> None:
            ...

        def __call__(self, memory_block: MemoryBlock) -> "ImageDecoderOutput":
            ...

else:
    from fairseq2n.bindings.data.image import ImageDecoder as ImageDecoder

    def _set_module_name() -> None:
        for t in [ImageDecoder]:
            t.__module__ = __name__

    _set_module_name()


class ImageDecoderOutput(TypedDict):
    bit_depth: float
    color_type: float
    channels: float
    height: float
    width: float
    image: Tensor
