# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Optional, TypedDict

from fairseq2 import _DOC_MODE
from fairseq2.memory import MemoryBlock
from fairseq2.typing import Device

if TYPE_CHECKING or _DOC_MODE:

    class PNGDecoder:
        def __init__(
            self,
            device: Optional[Device] = None,
            pin_memory: bool = False,
        ) -> None:
            ...

        def __call__(self, memory_block: MemoryBlock) -> "PNGDecoderOutput":
            ...

else:
    from fairseq2n.bindings.data.png import PNGDecoder as PNGDecoder

    def _set_module_name() -> None:
        for t in [PNGDecoder]:
            t.__module__ = __name__

    _set_module_name()


class PNGDecoderOutput(TypedDict):
    format: int
