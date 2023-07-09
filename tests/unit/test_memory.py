# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from array import array

import pytest

from fairseq2.memory import MemoryBlock


class TestMemoryBlock:
    def test_init_works_when_input_buffer_is_shared(self) -> None:
        arr = array("B", [0, 1, 2, 3])

        block = MemoryBlock(arr)

        arr[0] = 4

        del arr

        view = memoryview(block)  # type: ignore[arg-type]

        assert view.itemsize == 1
        assert view.ndim == 1
        assert view.shape == (4,)
        assert view.strides == (1,)

        assert view.tolist() == [4, 1, 2, 3]

    def test_init_works_when_input_buffer_is_shared_and_is_of_type_float(self) -> None:
        arr = array("f", [0.2, 0.4, 0.6])

        block = MemoryBlock(arr)

        del arr

        view = memoryview(block)  # type: ignore[arg-type]

        assert view.itemsize == 1
        assert view.ndim == 1
        assert view.shape == (12,)
        assert view.strides == (1,)

        view = view.cast("f")

        assert view.tolist() == pytest.approx([0.2, 0.4, 0.6])

    def test_init_works_when_copy_is_true(self) -> None:
        arr = array("B", [0, 1, 2, 3])

        block = MemoryBlock(arr, copy=True)

        arr[0] = 4

        view = memoryview(block)  # type: ignore[arg-type]

        assert view.tolist() == [0, 1, 2, 3]

    def test_pickle_works(self) -> None:
        arr = array("B", [0, 1, 2, 3])

        block = MemoryBlock(arr, copy=True)

        # We require pickle protocol 5 or higher since `MemoryBlock` uses
        # the `PickleBuffer` API for efficient memory sharing.
        dump = pickle.dumps(block, protocol=5)

        del block

        block = pickle.loads(dump)

        assert memoryview(block).tolist() == [0, 1, 2, 3]  # type: ignore[arg-type]
