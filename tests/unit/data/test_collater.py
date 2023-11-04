# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.nn.functional import pad

from fairseq2.data import CollateOptionsOverride, Collater
from tests.common import assert_close, assert_equal, device


class TestCollater:
    def test_call_works_when_input_has_only_non_composite_types(self) -> None:
        # fmt: off
        bucket = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        # fmt: on

        collater = Collater()

        assert collater(bucket) == [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

    def test_call_works_when_input_has_lists(self) -> None:
        # fmt: off
        bucket = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 0], [1, 2]]
        ]

        collater = Collater()

        assert collater(bucket) == [
            [[1, 5, 9], [2, 6, 0]], [[3, 7, 1], [4, 8, 2]]
        ]
        # fmt: on

    def test_call_works_when_input_has_dicts(self) -> None:
        # fmt: off
        bucket = [
            [{"foo1": 0, "foo2": 1}, {"foo3": 2, "foo4": 3}],
            [{"foo1": 4, "foo2": 5}, {"foo3": 6, "foo4": 7}],
            [{"foo1": 8, "foo2": 9}, {"foo3": 0, "foo4": 1}],
        ]
        # fmt: on

        collater = Collater()

        assert collater(bucket) == [
            {"foo1": [0, 4, 8], "foo2": [1, 5, 9]},
            {"foo3": [2, 6, 0], "foo4": [3, 7, 1]},
        ]

    def test_call_works_when_input_has_tensors(self) -> None:
        bucket = [
            torch.full((4,), 0.0, device=device, dtype=torch.float32),
            torch.full((4,), 1.0, device=device, dtype=torch.float32),
            torch.full((4,), 2.0, device=device, dtype=torch.float32),
        ]

        collater = Collater()

        expected_tensor = torch.tensor(
            [0.0, 1.0, 2.0], device=device, dtype=torch.float32
        )

        expected_tensor = expected_tensor.unsqueeze(-1).expand(-1, 4)

        assert_close(collater(bucket), expected_tensor)

    @pytest.mark.parametrize(
        "pad_to_multiple,pad_size", [(1, 0), (2, 0), (3, 2), (8, 4)]
    )
    def test_call_works_when_input_has_sequence_tensors(
        self, pad_to_multiple: int, pad_size: int
    ) -> None:
        bucket = [
            torch.full((4, 2), 0, device=device, dtype=torch.int64),
            torch.full((4, 2), 1, device=device, dtype=torch.int64),
            torch.full((4, 2), 2, device=device, dtype=torch.int64),
        ]

        collater = Collater(pad_value=3, pad_to_multiple=pad_to_multiple)

        output = collater(bucket)

        expected_seqs = torch.tensor(
            [
                [[0, 0], [0, 0], [0, 0], [0, 0]],
                [[1, 1], [1, 1], [1, 1], [1, 1]],
                [[2, 2], [2, 2], [2, 2], [2, 2]],
            ],
            device=device,
            dtype=torch.int64,
        )

        expected_seqs = pad(expected_seqs, (0, 0, 0, pad_size), value=3)

        expected_seq_lens = torch.tensor([4, 4, 4], device=device, dtype=torch.int64)

        assert_close(output["seqs"], expected_seqs)
        assert_equal(output["seq_lens"], expected_seq_lens)

        assert output["is_ragged"] == (pad_to_multiple > 2)

    def test_call_works_when_input_has_ragged_sequence_tensors(self) -> None:
        bucket = [
            {"foo1": torch.full((4, 2), 0, device=device, dtype=torch.int64)},
            {"foo1": torch.full((2, 2), 1, device=device, dtype=torch.int64)},
            {"foo1": torch.full((3, 2), 2, device=device, dtype=torch.int64)},
        ]

        collater = Collater(pad_value=3, pad_to_multiple=3)

        output = collater(bucket)

        expected_seqs = torch.tensor(
            [
                [[0, 0], [0, 0], [0, 0], [0, 0], [3, 3], [3, 3]],
                [[1, 1], [1, 1], [3, 3], [3, 3], [3, 3], [3, 3]],
                [[2, 2], [2, 2], [2, 2], [3, 3], [3, 3], [3, 3]],
            ],
            device=device,
            dtype=torch.int64,
        )

        expected_seq_lens = torch.tensor([4, 2, 3], device=device, dtype=torch.int64)

        assert_close(output["foo1"]["seqs"], expected_seqs)
        assert_equal(output["foo1"]["seq_lens"], expected_seq_lens)

        assert output["foo1"]["is_ragged"] == True

    def test_call_works_when_options_are_overriden(self) -> None:
        # fmt: off
        bucket = [
            {"foo1": torch.full((4,2), 0, device=device, dtype=torch.int32), "foo2": torch.full((4,2), 0, device=device, dtype=torch.int64)},
            {"foo1": torch.full((2,2), 1, device=device, dtype=torch.int32), "foo2": torch.full((2,2), 1, device=device, dtype=torch.int64)},
            {"foo1": torch.full((3,2), 2, device=device, dtype=torch.int32), "foo2": torch.full((3,2), 2, device=device, dtype=torch.int64)},
        ]
        # fmt: on

        collater = Collater(
            pad_value=1,
            overrides=[CollateOptionsOverride("foo1", pad_value=3, pad_to_multiple=3)],
        )

        output = collater(bucket)

        expected_seqs1 = torch.tensor(
            [
                [[0, 0], [0, 0], [0, 0], [0, 0], [3, 3], [3, 3]],
                [[1, 1], [1, 1], [3, 3], [3, 3], [3, 3], [3, 3]],
                [[2, 2], [2, 2], [2, 2], [3, 3], [3, 3], [3, 3]],
            ],
            device=device,
            dtype=torch.int32,
        )

        expected_seqs2 = torch.tensor(
            [
                [[0, 0], [0, 0], [0, 0], [0, 0]],
                [[1, 1], [1, 1], [1, 1], [1, 1]],
                [[2, 2], [2, 2], [2, 2], [1, 1]],
            ],
            device=device,
            dtype=torch.int64,
        )

        expected_seq_lens = torch.tensor([4, 2, 3], device=device, dtype=torch.int64)

        assert_close(output["foo1"]["seqs"], expected_seqs1)
        assert_close(output["foo2"]["seqs"], expected_seqs2)

        assert_equal(output["foo1"]["seq_lens"], expected_seq_lens)
        assert_equal(output["foo2"]["seq_lens"], expected_seq_lens)

        assert output["foo1"]["is_ragged"] == True
        assert output["foo2"]["is_ragged"] == True

    def test_call_works_when_input_has_composite_elements(self) -> None:
        # fmt: off
        bucket = [
            {"foo1": [0, 1, 2], "foo2": {"subfoo1": 0, "subfoo2": torch.full((4,), 0.0, device=device)}, "foo3": 1},
            {"foo1": [3, 4, 5], "foo2": {"subfoo1": 1, "subfoo2": torch.full((4,), 1.0, device=device)}, "foo3": 2},
            {"foo1": [6, 7, 8], "foo2": {"subfoo1": 2, "subfoo2": torch.full((4,), 2.0, device=device)}, "foo3": 3},
        ]
        # fmt: on

        collater = Collater()

        output = collater(bucket)

        expected_tensor = torch.tensor(
            [0.0, 1.0, 2.0], device=device, dtype=torch.float32
        )

        expected_tensor = expected_tensor.unsqueeze(-1).expand(-1, 4)

        assert_close(output["foo2"]["subfoo2"], expected_tensor)

        del output["foo2"]["subfoo2"]

        # fmt: off
        assert output == {"foo1": [[0, 3, 6], [1, 4, 7], [2, 5, 8]], "foo2": {"subfoo1": [0, 1, 2]}, "foo3": [1, 2, 3]}
        # fmt: on

    def test_call_raises_error_when_input_is_empty(self) -> None:
        collater = Collater()

        with pytest.raises(
            ValueError,
            match=r"^The bucket must contain at least one element, but is empty instead\.$",
        ):
            collater([])

    def test_call_works_when_input_is_not_a_bucket(self) -> None:
        bucket = {"foo1": 1}

        collater = Collater()

        assert collater(bucket) == {"foo1": [1]}
        assert collater(1) == [1]

    def test_call_raises_error_when_items_have_different_types(self) -> None:
        bucket = [1, "foo", 2]

        collater = Collater()

        with pytest.raises(
            ValueError,
            match=r"^The bucket item 1 must be of type `int`, but is of type `string` instead\.$",
        ):
            collater(bucket)

        bucket = [[1, 2], "foo", [3, 4]]

        with pytest.raises(
            ValueError,
            match=r"^The bucket item 1 must be of type `list`, but is of type `string` instead\.$",
        ):
            collater(bucket)

        bucket = [{"foo1": 0}, "foo", [3, 4]]

        with pytest.raises(
            ValueError,
            match=r"^The bucket item 1 must be of type `dict`, but is of type `string` instead\.$",
        ):
            collater(bucket)

        bucket = [[1, 2], [3, "b"], [5, 6]]

        with pytest.raises(
            ValueError,
            match=r"^The element at path '\[1\]' in the bucket item 1 must be of type `int`, but is of type `string` instead\.$",
        ):
            collater(bucket)

        bucket = [torch.empty(()), 4, "a"]

        with pytest.raises(
            ValueError,
            match=r"^The bucket item 1 must be of type `torch\.Tensor`, but is of type `int` instead\.$",
        ):
            collater(bucket)

    def test_call_raises_error_when_items_have_different_elements(self) -> None:
        bucket = [{"foo1": 0}, {"foo2": 1}, {"foo1:": 2}]

        collater = Collater()

        with pytest.raises(
            ValueError,
            match=r"^The bucket item 1 does not have an element at path 'foo1'\.$",
        ):
            collater(bucket)

    def test_call_raises_error_when_lists_have_different_lengths(self) -> None:
        # fmt: off
        bucket1 = [
            [1, 2, 3],
            [4, 5],
            [7, 8, 9]
        ]
        # fmt: on

        collater = Collater()

        with pytest.raises(
            ValueError,
            match=r"^The bucket item 1 must have a length of 3, but has a length of 2 instead\.$",
        ):
            collater(bucket1)

        # fmt: off
        bucket2 = [
            [[1, 2], [3, 4]],
            [[5, 6], [7]],
            [[9, 0], [1, 2]],
        ]
        # fmt: on

        with pytest.raises(
            ValueError,
            match=r"^The `list` at path '\[1\]' in the bucket item 1 must have a length of 2, but has a length of 1 instead\.$",
        ):
            collater(bucket2)

    def test_call_raises_error_when_dicts_have_different_lengths(self) -> None:
        # fmt: off
        bucket1 = [
            {"foo1": 0, "foo2": 1, "foo3": 2},
            {"foo1": 1, "foo2": 2},
            {"foo1": 2, "foo2": 3, "foo3": 4},
        ]
        # fmt: on

        collater = Collater()

        with pytest.raises(
            ValueError,
            match=r"^The bucket item 1 must have a length of 3, but has a length of 2 instead\.$",
        ):
            collater(bucket1)

        # fmt: off
        bucket2 = [
            [{"foo1": 0, "foo2": 1, "foo3": 2}],
            [{"foo1": 1, "foo2": 2}],
            [{"foo1": 2, "foo2": 3, "foo3": 4}],
        ]
        # fmt: on

        with pytest.raises(
            ValueError,
            match=r"^The `dict` at path '\[0\]' in the bucket item 1 must have a length of 3, but has a length of 2 instead\.$",
        ):
            collater(bucket2)

    def test_call_raises_error_when_tensors_have_different_shapes(self) -> None:
        bucket1 = [
            torch.zeros((4,), device=device),
            torch.zeros((3,), device=device),
            torch.zeros((4,), device=device),
        ]

        collater = Collater()

        with pytest.raises(
            ValueError,
            match=r"^The tensors in the bucket cannot be stacked. See nested exception for details\.$",
        ):
            collater(bucket1)

        bucket2 = [
            [torch.zeros((4,), device=device)],
            [torch.zeros((3,), device=device)],
            [torch.zeros((4,), device=device)],
        ]

        collater = Collater()

        with pytest.raises(
            ValueError,
            match=r"^The tensors at path '\[0\]' in the bucket cannot be stacked. See nested exception for details\.$",
        ):
            collater(bucket2)

    def test_init_raises_error_when_pad_value_is_none_and_pad_to_multiple_is_greater_than_1(
        self,
    ) -> None:
        with pytest.raises(
            ValueError,
            match=r"^`pad_value` must be set when `pad_to_multiple` is greater than 1\.$",
        ):
            Collater(pad_to_multiple=4)

        with pytest.raises(
            ValueError,
            match=r"^`pad_value` of the selector 'foo' must be set when `pad_to_multiple` is greater than 1\.$",
        ):
            Collater(overrides=[CollateOptionsOverride("foo", pad_to_multiple=2)])
