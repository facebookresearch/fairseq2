# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from array import array

import pytest

from fairseq2.data import MemoryBlock


class TestMemoryBlock:
    def test_returns_shared_byte_buffer(self) -> None:
        arr = array("B", [0, 1, 2, 3])

        blk = MemoryBlock(arr)

        arr[0] = 4

        del arr

        mem = memoryview(blk)  # type: ignore[arg-type]

        assert mem.itemsize == 1
        assert mem.ndim == 1
        assert mem.shape == (4,)
        assert mem.strides == (1,)

        assert mem.tolist() == [4, 1, 2, 3]

    def test_returns_shared_float_buffer(self) -> None:
        arr = array("f", [0.2, 0.4, 0.6])

        blk = MemoryBlock(arr)

        del arr

        mem = memoryview(blk)  # type: ignore[arg-type]

        assert mem.itemsize == 1
        assert mem.ndim == 1
        assert mem.shape == (12,)
        assert mem.strides == (1,)

        mem = mem.cast("f")

        assert mem.tolist() == pytest.approx([0.2, 0.4, 0.6])

    def test_returns_non_shared_by_buffer_if_copy_is_true(self) -> None:
        arr = array("B", [0, 1, 2, 3])

        blk = MemoryBlock(arr, copy=True)

        arr[0] = 4

        mem = memoryview(blk)  # type: ignore[arg-type]

        assert mem.tolist() == [0, 1, 2, 3]
