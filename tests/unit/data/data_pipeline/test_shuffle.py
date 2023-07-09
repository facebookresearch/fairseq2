# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import islice

import pytest
import torch

from fairseq2.data import read_sequence
from tests.common import tmp_rng_seed


class TestShuffleOp:
    def test_op_works(self) -> None:
        cpu_device = torch.device("cpu")

        seq = list(range(1, 10))

        pipeline = read_sequence(seq).shuffle(100).and_return()

        for _ in range(2):
            with tmp_rng_seed(cpu_device, seed=2):
                assert list(pipeline) == [7, 1, 3, 2, 6, 5, 8, 4, 9]

                pipeline.reset()

        pipeline = read_sequence(seq).shuffle(0).and_return()

        for _ in range(2):
            with tmp_rng_seed(cpu_device, seed=2):
                assert list(pipeline) == [7, 1, 3, 2, 6, 5, 8, 4, 9]

                pipeline.reset()

        pipeline = read_sequence(seq).shuffle(3).and_return()

        for _ in range(2):
            with tmp_rng_seed(cpu_device, seed=2):
                assert list(pipeline) == [1, 3, 5, 2, 6, 7, 4, 9, 8]

                pipeline.reset()

        pipeline = read_sequence(seq).shuffle(1).and_return()

        for _ in range(2):
            with tmp_rng_seed(cpu_device, seed=2):
                assert list(pipeline) == [1, 2, 3, 4, 5, 6, 7, 8, 9]

                pipeline.reset()

    @pytest.mark.parametrize("window", [10, 100, 1000])
    def test_op_saves_and_restores_its_state(self, window: int) -> None:
        cpu_device = torch.device("cpu")

        seq = list(range(5000))

        pipeline1 = read_sequence(seq).shuffle(window).and_return()
        pipeline2 = read_sequence(seq).shuffle(window).and_return()

        with tmp_rng_seed(cpu_device, seed=2):
            expected_output1 = list(islice(pipeline1, 4000))

        with tmp_rng_seed(cpu_device, seed=3):
            expected_output2 = list(islice(pipeline1, 1000))

        with tmp_rng_seed(cpu_device, seed=2):
            assert list(islice(pipeline2, 4000)) == expected_output1

        state_dict = pipeline2.state_dict()

        with tmp_rng_seed(cpu_device, seed=3):
            assert list(islice(pipeline2, 1000)) == expected_output2

        pipeline2.load_state_dict(state_dict)

        with tmp_rng_seed(cpu_device, seed=3):
            assert list(islice(pipeline2, 1000)) == expected_output2

        pipeline2.reset()
        pipeline2.load_state_dict(state_dict)

        with tmp_rng_seed(cpu_device, seed=3):
            assert list(islice(pipeline2, 1000)) == expected_output2

        state_dict = pipeline2.state_dict()

        with pytest.raises(StopIteration):
            next(iter(pipeline2))

        pipeline2.reset()

        pipeline2.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline2))

    def test_record_reload_position_works_as_expected_with_no_strict(self) -> None:
        seq = list(range(100))

        pipeline = read_sequence(seq).shuffle(80, strict=False).and_return()

        # Do one dummy iteration to force to fill the buffer.
        next(iter(pipeline))

        state_dict = pipeline.state_dict()

        pipeline.load_state_dict(state_dict)

        assert min(pipeline) == 81
