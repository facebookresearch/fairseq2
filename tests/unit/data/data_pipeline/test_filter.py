# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from fairseq2.data.data_pipeline import read_sequence


class TestFilterOp:
    def test_op_works(self) -> None:
        def fn(d: int) -> bool:
            return d % 2 == 1

        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        pipeline = read_sequence(seq).filter(fn).and_return()

        for _ in range(2):
            assert list(pipeline) == [1, 3, 5, 7, 9]

            pipeline.reset()

    def test_op_raises_nested_error_when_callable_fails(self) -> None:
        def fn(d: int) -> bool:
            if d == 3:
                raise ValueError("filter error")

            return True

        pipeline = read_sequence([1, 2, 3, 4]).filter(fn).and_return()

        with pytest.raises(ValueError) as exc_info:
            for d in pipeline:
                pass

        assert str(exc_info.value) == "filter error"
