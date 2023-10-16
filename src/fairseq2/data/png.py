# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Optional, TypedDict, Union

from torch import Tensor
from typing_extensions import NotRequired

from fairseq2 import _DOC_MODE
from fairseq2.memory import MemoryBlock
from fairseq2.typing import DataType, Device

if TYPE_CHECKING or _DOC_MODE:

    class ImageDecoder:
        def __init__(
            self,
            dtype: Optional[DataType] = None,
            device: Optional[Device] = None,
            pin_memory: bool = False,
        ) -> None:
            ...

        def __call__(self, memory_block: MemoryBlock) -> "ImageDecoderOutput":
            ...

else:
    from fairseq2n.bindings.data.png import ImageDecoder as ImageDecoder

    def _set_module_name() -> None:
        for t in [ImageDecoder]:
            t.__module__ = __name__

    _set_module_name()

class ImageDecoderOutput(TypedDict):
    format: int
