# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from array import array
from typing import TYPE_CHECKING, Optional, Union, overload

from fairseq2n import DOC_MODE
from typing_extensions import TypeAlias

Buffer: TypeAlias = Union[bytes, bytearray, memoryview, array]

if TYPE_CHECKING or DOC_MODE:

    class MemoryBlock:
        """Represents a contiguous block of read-only memory."""

        @overload
        def __init__(self) -> None:
            ...

        @overload
        def __init__(self, buffer: Buffer, copy: bool = False) -> None:
            ...

        def __init__(self, buffer: Optional[Buffer] = None, copy: bool = False) -> None:
            """
            :param buffer:
                An object that supports the Python buffer protocol.
            :param copy:
                If ``True``, copies ``buffer``.
            """

        def __len__(self) -> int:
            ...

        def __bytes__(self) -> bytes:
            ...

else:
    from fairseq2n.bindings.memory import MemoryBlock as MemoryBlock

    MemoryBlock.__module__ = __name__
