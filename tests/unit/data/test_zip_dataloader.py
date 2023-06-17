# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Union

import torch
from torch import Tensor

import fairseq2.data
from tests.common import assert_equal
from tests.unit.data.test_dataloader import assert_eq_twice


def assert_equal_tensor_list(
    actual: Union[Tensor, List[Any]], expected: Union[Tensor, List[Any]]
) -> None:
    """Assert equality of embeded list of tensors"""
    assert type(actual) == type(expected)

    if isinstance(actual, Tensor):
        assert_equal(actual, expected)
    elif isinstance(actual, List):
        for i in range(len(actual)):
            assert_equal_tensor_list(actual[i], expected[i])
    else:
        raise ValueError(f"{type(actual)} not supported.")


def test_zip_2dl() -> None:
    X1 = fairseq2.data.read_sequence([1, 2, 3]).and_return()
    X2 = fairseq2.data.read_sequence([4, 5, 6]).and_return()

    zipped = fairseq2.data.zip_data_pipelines([X1, X2]).and_return()
    expected = [[1, 4], [2, 5], [3, 6]]

    assert_eq_twice(zipped, expected)


def test_zip_3dl() -> None:
    X1 = fairseq2.data.read_sequence([1, 2, 3]).and_return()
    X2 = fairseq2.data.read_sequence([4, 5, 6]).and_return()
    X3 = fairseq2.data.read_sequence([7, 8, 9]).and_return()

    zipped = fairseq2.data.zip_data_pipelines([X1, X2, X3]).and_return()
    expected = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

    assert_eq_twice(zipped, expected)


def test_zip_2dl_different_size() -> None:
    X1 = fairseq2.data.read_sequence([1, 2, 3, 4]).and_return()
    X2 = fairseq2.data.read_sequence([5, 6]).and_return()

    zipped = fairseq2.data.zip_data_pipelines([X1, X2]).and_return()
    expected = [[1, 5], [2, 6]]

    assert_eq_twice(zipped, expected)


def test_zip_tensors() -> None:
    X1 = (
        fairseq2.data.read_sequence([1, 2, 3])
        .map(lambda l: torch.zeros(l))
        .and_return()
    )
    X2 = (
        fairseq2.data.read_sequence([1, 2, 3]).map(lambda l: torch.ones(l)).and_return()
    )

    zipped = fairseq2.data.zip_data_pipelines([X1, X2]).and_return()
    expected = [
        [torch.zeros(1), torch.ones(1)],
        [torch.zeros(2), torch.ones(2)],
        [torch.zeros(3), torch.ones(3)],
    ]

    assert_equal_tensor_list(list(zipped), expected)


def test_zip_batch() -> None:
    X1 = fairseq2.data.read_sequence([10, 11, 12, 13, 14, 15]).and_return()
    X2 = fairseq2.data.read_sequence([20, 21, 22, 23, 24, 25]).and_return()
    X3 = fairseq2.data.read_sequence([30, 31, 32, 33, 34, 35]).and_return()

    zipped = fairseq2.data.zip_data_pipelines([X1, X2, X3]).batch(4).and_return()
    expected = [
        [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
        [[14, 15], [24, 25], [34, 35]],
    ]

    assert_eq_twice(zipped, expected)


def test_zip_batch_tensor() -> None:
    X1 = fairseq2.data.read_sequence(
        [
            torch.tensor([11, 11, 11], dtype=torch.int32),
            torch.tensor([12, 12, 12], dtype=torch.int32),
            torch.tensor([13, 13, 13], dtype=torch.int32),
        ]
    ).and_return()
    X2 = fairseq2.data.read_sequence(
        [
            torch.tensor([21, 21, 21], dtype=torch.int32),
            torch.tensor([22, 22, 22], dtype=torch.int32),
            torch.tensor([23, 23, 23], dtype=torch.int32),
        ]
    ).and_return()
    X3 = fairseq2.data.read_sequence(
        [
            torch.tensor([31, 31, 31], dtype=torch.int32),
            torch.tensor([32, 32, 32], dtype=torch.int32),
            torch.tensor([33, 33, 33], dtype=torch.int32),
        ]
    ).and_return()

    zipped = fairseq2.data.zip_data_pipelines([X1, X2, X3]).batch(2).and_return()
    expected = [
        [
            torch.tensor([[11, 11, 11], [12, 12, 12]], dtype=torch.int32),
            torch.tensor([[21, 21, 21], [22, 22, 22]], dtype=torch.int32),
            torch.tensor([[31, 31, 31], [32, 32, 32]], dtype=torch.int32),
        ],
        [
            torch.tensor([[13, 13, 13]], dtype=torch.int32),
            torch.tensor([[23, 23, 23]], dtype=torch.int32),
            torch.tensor([[33, 33, 33]], dtype=torch.int32),
        ],
    ]

    assert_equal_tensor_list(list(zipped), expected)


def test_zip_batch_tensor_padding() -> None:
    X1 = fairseq2.data.read_sequence(
        [
            torch.tensor([11, 11], dtype=torch.int32),
            torch.tensor([12, 12, 12], dtype=torch.int32),
            torch.tensor([13, 13, 13], dtype=torch.int32),
        ]
    ).and_return()
    X2 = fairseq2.data.read_sequence(
        [
            torch.tensor([21, 21], dtype=torch.int32),
            torch.tensor([22, 22, 22], dtype=torch.int32),
            torch.tensor([23, 23, 23], dtype=torch.int32),
        ]
    ).and_return()
    X3 = fairseq2.data.read_sequence(
        [
            torch.tensor([31, 31], dtype=torch.int32),
            torch.tensor([32, 32, 32], dtype=torch.int32),
            torch.tensor([33, 33, 33], dtype=torch.int32),
        ]
    ).and_return()

    zipped = (
        fairseq2.data.zip_data_pipelines([X1, X2, X3]).batch(2, pad_idx=0).and_return()
    )
    expected = [
        [
            torch.tensor([[11, 11, 0], [12, 12, 12]], dtype=torch.int32),
            torch.tensor([[21, 21, 0], [22, 22, 22]], dtype=torch.int32),
            torch.tensor([[31, 31, 0], [32, 32, 32]], dtype=torch.int32),
        ],
        [
            torch.tensor([[13, 13, 13]], dtype=torch.int32),
            torch.tensor([[23, 23, 23]], dtype=torch.int32),
            torch.tensor([[33, 33, 33]], dtype=torch.int32),
        ],
    ]

    assert_equal_tensor_list(list(zipped), expected)


def test_zip_batch_padding_multiple() -> None:
    X1 = fairseq2.data.read_sequence(
        [
            torch.tensor([11, 11], dtype=torch.int32),
            torch.tensor([12, 12, 12], dtype=torch.int32),
            torch.tensor([13, 13, 13], dtype=torch.int32),
        ]
    ).and_return()
    X2 = fairseq2.data.read_sequence(
        [
            torch.tensor([21, 21], dtype=torch.int32),
            torch.tensor([22, 22, 22], dtype=torch.int32),
            torch.tensor([23, 23, 23], dtype=torch.int32),
        ]
    ).and_return()
    X3 = fairseq2.data.read_sequence(
        [
            torch.tensor([31, 31], dtype=torch.int32),
            torch.tensor([32, 32, 32], dtype=torch.int32),
            torch.tensor([33, 33, 33], dtype=torch.int32),
        ]
    ).and_return()

    zipped = (
        fairseq2.data.zip_data_pipelines([X1, X2, X3])
        .batch(2, pad_idx=[1, 2, 3])
        .and_return()
    )
    expected = [
        [
            torch.tensor([[11, 11, 1], [12, 12, 12]], dtype=torch.int32),
            torch.tensor([[21, 21, 2], [22, 22, 22]], dtype=torch.int32),
            torch.tensor([[31, 31, 3], [32, 32, 32]], dtype=torch.int32),
        ],
        [
            torch.tensor([[13, 13, 13]], dtype=torch.int32),
            torch.tensor([[23, 23, 23]], dtype=torch.int32),
            torch.tensor([[33, 33, 33]], dtype=torch.int32),
        ],
    ]

    assert_equal_tensor_list(list(zipped), expected)
