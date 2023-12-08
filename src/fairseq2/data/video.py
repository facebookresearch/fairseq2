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

    class VideoDecoder:
        def __init__(
            self,
            dtype: Optional[DataType] = None,
            device: Optional[Device] = None,
            pin_memory: bool = False,
            get_pts_only: Optional[bool] = False,
            get_frames_only: Optional[bool] = False,
            width: Optional[int] = None,
            height: Optional[int] = None,
        ) -> None:
            ...

        def __call__(self, memory_block: MemoryBlock) -> "VideoDecoderOutput":
            ...

else:
    from fairseq2n.bindings.data.video import VideoDecoder as VideoDecoder

    def _set_module_name() -> None:
        for t in [VideoDecoder]:
            t.__module__ = __name__

    _set_module_name()


class VideoDecoderOutput(TypedDict):
    video: Tensor
