# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch

from fairseq2.data.data_pipeline import read_sequence
from tests.common import assert_equal


class TestPackOp:
    def test_op_works(self) -> None:
        seqs = [
            torch.tensor([1, 2, 3, 0]),
            torch.tensor([5, 6, 0]),
            torch.tensor([9, 1, 2, 3, 0]),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([4, 0]),
        ]

        pipeline = (
            read_sequence(seqs).pack(num_elements=3, max_seq_len=32, drop_remainder=False).and_return()  # fmt: skip
        )

        expected_examples = [
            (torch.tensor([1, 2, 3]), [3]),
            (torch.tensor([3, 0, 5]), [2, 1]),
            (torch.tensor([5, 6, 0]), [3]),
            (torch.tensor([9, 1, 2]), [3]),
            (torch.tensor([2, 3, 0]), [3]),
            (torch.tensor([4, 0, 0]), [2]),
        ]

        for _ in range(2):
            examples = list(pipeline)

            assert len(examples) == len(expected_examples)

            for example, expected_example in zip(examples, expected_examples):
                expected_seqs, expected_seq_lens = expected_example

                assert_equal(example["seqs"], expected_seqs)

                assert example["seq_lens"] == expected_seq_lens

            pipeline.reset()

    def test_op_works_when_drop_remainder_is_true(self) -> None:
        seqs = [
            torch.tensor([1, 2, 3, 0]),
            torch.tensor([5, 6, 0]),
            torch.tensor([9, 1, 2, 3, 0]),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([4, 0]),
        ]

        pipeline = (
            read_sequence(seqs).pack(num_elements=3, max_seq_len=32, drop_remainder=True).and_return()  # fmt: skip
        )

        expected_examples = [
            (torch.tensor([1, 2, 3]), [3]),
            (torch.tensor([3, 0, 5]), [2, 1]),
            (torch.tensor([5, 6, 0]), [3]),
            (torch.tensor([9, 1, 2]), [3]),
            (torch.tensor([2, 3, 0]), [3]),
        ]

        for _ in range(2):
            examples = list(pipeline)

            assert len(examples) == len(expected_examples)

            for example, expected_example in zip(examples, expected_examples):
                expected_seqs, expected_seq_lens = expected_example

                assert_equal(example["seqs"], expected_seqs)

                assert example["seq_lens"] == expected_seq_lens

            pipeline.reset()

    def test_op_works_when_seq_is_longer_than_max(self) -> None:
        seqs = [
            torch.tensor([1, 2, 3, 4, 5, 0]),
            torch.tensor([6, 7, 8, 0]),
            torch.tensor([9, 1, 2, 3, 0]),
            torch.tensor([4, 0]),
        ]

        pipeline = (
            read_sequence(seqs).pack(num_elements=6, max_seq_len=3, drop_remainder=False).and_return()  # fmt: skip
        )

        expected_examples = [
            (torch.tensor([1, 2, 3, 4, 5, 0]), [3, 3]),
            (torch.tensor([6, 7, 8, 0, 9, 1]), [3, 1, 2]),
            (torch.tensor([1, 2, 3, 0, 4, 0]), [3, 1, 2]),
        ]

        for _ in range(2):
            examples = list(pipeline)

            assert len(examples) == len(expected_examples)

            for example, expected_example in zip(examples, expected_examples):
                expected_seqs, expected_seq_lens = expected_example

                assert_equal(example["seqs"], expected_seqs)

                assert example["seq_lens"] == expected_seq_lens

            pipeline.reset()

    def test_op_works_when_truncate_is_false(self) -> None:
        seqs = [
            torch.tensor([1, 2, 3, 0]),
            torch.tensor([5, 6, 0]),
            torch.tensor([9, 1, 2, 3, 0]),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([4, 1, 0]),
        ]

        # `drop_remainder` should have no effect when we do not truncate.
        pipeline = (
            read_sequence(seqs).pack(num_elements=7, max_seq_len=32, truncate=False, drop_remainder=True).and_return()  # fmt: skip
        )

        expected_examples = [
            (torch.tensor([1, 2, 3, 0, 5, 6, 0]), [4, 3]),
            (torch.tensor([9, 1, 2, 3, 0, 0, 0]), [5]),
            (torch.tensor([4, 1, 0, 0, 0, 0, 0]), [3]),
        ]

        for _ in range(2):
            examples = list(pipeline)

            assert len(examples) == len(expected_examples)

            for example, expected_example in zip(examples, expected_examples):
                expected_seqs, expected_seq_lens = expected_example

                assert_equal(example["seqs"], expected_seqs)

                assert example["seq_lens"] == expected_seq_lens

            pipeline.reset()

    def test_op_saves_and_restores_its_state(self) -> None:
        seqs = [
            torch.tensor([1, 2, 3, 0]),
            torch.tensor([5, 6, 0]),
            torch.tensor([9, 1, 2, 3, 0]),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([4, 0]),
        ]

        pipeline = (
            read_sequence(seqs).pack(num_elements=3, max_seq_len=32, drop_remainder=False).and_return()  # fmt: skip
        )

        it = iter(pipeline)

        # Move to the second example.
        next(it)

        d = next(it)

        assert_equal(d["seqs"], torch.tensor([3, 0, 5]))

        state_dict = pipeline.state_dict()

        # Read a few examples before we roll back.
        next(it)

        d = next(it)

        assert_equal(d["seqs"], torch.tensor([9, 1, 2]))

        # Expected to roll back to the second example.
        pipeline.load_state_dict(state_dict)

        # Move to EOD.
        for _ in range(3):
            next(it)

        d = next(it)

        assert_equal(d["seqs"], torch.tensor([4, 0, 0]))

        state_dict = pipeline.state_dict()

        pipeline.reset()

        # Expected to be EOD.
        pipeline.load_state_dict(state_dict)

        with pytest.raises(StopIteration):
            next(iter(pipeline))
