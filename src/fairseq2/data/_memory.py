# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, TypeAlias, final, overload

from fairseq2n import DOC_MODE

Buffer: TypeAlias = bytes | bytearray | memoryview | array  # type: ignore[type-arg]

if TYPE_CHECKING or DOC_MODE:

    @final
    class MemoryBlock:
        """Represents a contiguous block of read-only memory."""

        @overload
        def __init__(self) -> None: ...

        @overload
        def __init__(self, buffer: Buffer, copy: bool = False) -> None: ...

        def __init__(self, buffer: Buffer | None = None, copy: bool = False) -> None:
            """
            :param buffer:
                An object that supports the Python buffer protocol.
            :param copy:
                If ``True``, copies ``buffer``.
            """

        def __len__(self) -> int: ...

        def __bytes__(self) -> bytes: ...

else:
    from fairseq2n.bindings.memory import MemoryBlock as MemoryBlock  # noqa: F401
