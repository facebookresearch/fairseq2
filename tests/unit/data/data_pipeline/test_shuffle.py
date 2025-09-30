# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from itertools import islice

import pytest

from fairseq2.data.data_pipeline import read_sequence


class TestShuffleOp:
    def test_op_works(self) -> None:
        seq = list(range(1, 10))

        # Shuffle 100.
        pipeline = read_sequence(seq).shuffle(100, seed=1234).and_return()

        for _ in range(2):
            assert list(pipeline) == [8, 9, 3, 7, 5, 4, 2, 6, 1]

            pipeline.reset(reset_rng=True)

        list(pipeline)

        pipeline.reset(reset_rng=False)

        # We haven't reset the seed. The list should be different this time.
        assert list(pipeline) != [8, 9, 3, 7, 5, 4, 2, 6, 1]

        # Shuffle the whole list.
        pipeline = read_sequence(seq).shuffle(0, seed=1234).and_return()

        for _ in range(2):
            assert list(pipeline) == [8, 9, 3, 7, 5, 4, 2, 6, 1]

            pipeline.reset(reset_rng=True)

        # Shuffle 2.
        pipeline = read_sequence(seq).shuffle(4, seed=1234).and_return()

        for _ in range(2):
            assert list(pipeline) == [2, 1, 3, 4, 5, 7, 8, 6, 9]

            pipeline.reset(reset_rng=True)

        # Shuffle 1.
        pipeline = read_sequence(seq).shuffle(1, seed=1234).and_return()

        for _ in range(2):
            assert list(pipeline) == [1, 2, 3, 4, 5, 6, 7, 8, 9]

            pipeline.reset(reset_rng=True)

    def test_op_saves_its_state_after_internal_buffer_is_emptied(self) -> None:
        class Foo:
            pass

        # Deliberately use an opaque Python object to ensure that it cannot be
        # saved in case the buffer is not emptied correctly.
        seq = [Foo()] * 10

        pipeline = read_sequence(seq).shuffle(40).and_return()

        # Exhaust the whole pipeline so that the internal shuffle buffer is
        # emptied.
        _ = list(pipeline)

        # Must not fail.
        pipeline.state_dict()

    @pytest.mark.parametrize("window", [10, 100, 1000])
    def test_op_saves_and_restores_its_state(self, window: int) -> None:
        seq = list(range(2000))

        pipeline = read_sequence(seq).shuffle(window, seed=1234).and_return()

        it = iter(pipeline)

        for _ in range(1000):
            next(it)

        state_dict = pipeline.state_dict()

        expected_output = list(islice(pipeline, 1000))

        pipeline.reset()

        pipeline.load_state_dict(state_dict)

        assert list(islice(pipeline, 1000)) == expected_output

        with pytest.raises(StopIteration):
            next(iter(pipeline))

    def test_record_reload_position_works_as_expected_with_no_strict(self) -> None:
        seq = list(range(100))

        pipeline = read_sequence(seq).shuffle(80).and_return()

        # Do one dummy iteration to force to fill the buffer.
        next(iter(pipeline))

        state_dict = pipeline.state_dict(strict=False)

        pipeline.reset()

        pipeline.load_state_dict(state_dict)

        assert min(pipeline) == 81
