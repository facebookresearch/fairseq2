# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from fairseq2.data import read_sequence


class TestFlattenOp:
    def test_op_works_without_selector(self) -> None:
        # Test flattening a list of lists without a selector
        seq = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

        pipeline = read_sequence(seq).flatten().and_return()

        # Expected output is a flattened list
        expected_output = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        for _ in range(2):
            assert list(pipeline) == expected_output

            pipeline.reset()

    def test_op_works_when_pipeline_is_empty(self) -> None:
        # Test flattening an empty sequence
        pipeline = read_sequence([]).flatten().and_return()

        for _ in range(2):
            assert list(pipeline) == []

            pipeline.reset()

    def test_op_works_with_empty_lists(self) -> None:
        # Test behavior with empty lists
        seq = [[1, 2, 3], [], [4, 5], [], []]

        pipeline = read_sequence(seq).flatten().and_return()

        # Expected output is the flattened non-empty lists
        expected_output = [1, 2, 3, 4, 5]

        for _ in range(2):
            assert list(pipeline) == expected_output

            pipeline.reset()

    def test_op_saves_and_restores_its_state(self) -> None:
        # Test state saving and restoration
        seq = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

        pipeline = read_sequence(seq).flatten().and_return()

        d = None

        it = iter(pipeline)

        # Move to the fifth example.
        for _ in range(5):
            d = next(it)

        assert d == 5

        state_dict = pipeline.state_dict()

        # Read a few examples before we roll back.
        for _ in range(2):
            d = next(it)

        assert d == 7

        # Expected to roll back to the fifth example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(4):
            d = next(it)

        assert d == 9

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline))

    def test_nested_flattening(self) -> None:
        # Test flattening nested structures
        seq = [[[1, 2], [3, 4]], [[5, 6]], [[7, 8, 9]]]

        # First flatten the outer lists
        pipeline = read_sequence(seq).flatten().and_return()
        
        # The first flatten gives us [[1, 2], [3, 4], [5, 6], [7, 8, 9]]
        intermediate = list(pipeline)
        
        pipeline.reset()
        
        # Now flatten the result again
        pipeline2 = read_sequence(intermediate).flatten().and_return()
        
        expected_output = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        assert list(pipeline2) == expected_output
