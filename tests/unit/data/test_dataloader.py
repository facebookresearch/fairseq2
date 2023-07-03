# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, List

import pytest
import torch

import fairseq2.data.text
from fairseq2.data import Collater

FILE = Path(__file__)
FILE_LINES = FILE.read_text().splitlines()


def test_read_text() -> None:
    dataloader = fairseq2.data.text.read_text(FILE, rtrim=True).and_return()
    assert_eq_twice(dataloader, FILE_LINES)


def test_read_text_skip_empty() -> None:
    dataloader = fairseq2.data.text.read_text(
        FILE, rtrim=True, skip_empty=True
    ).and_return()
    ACTUAL_LINES = [l for l in FILE_LINES if l]
    assert_eq_twice(dataloader, ACTUAL_LINES)


def test_bucket() -> None:
    dataloader = fairseq2.data.text.read_text(FILE, rtrim=True).bucket(8).and_return()
    dataloader_drop_remainder = (
        fairseq2.data.text.read_text(FILE, rtrim=True)
        .bucket(8, drop_remainder=True)
        .map(Collater())
        .and_return()
    )

    dataset = list(dataloader)
    dataset_drop_remainder = list(iter(dataloader_drop_remainder))
    assert [l for batch in dataset for l in batch] == FILE_LINES

    N = len(FILE_LINES)
    expected_batch_sizes = [8] * (N // 8)
    assert [len(batch) for batch in dataset_drop_remainder] == expected_batch_sizes

    if N % 8 != 0:
        expected_batch_sizes.append(N % 8)

    assert [len(batch) for batch in dataset] == expected_batch_sizes


def test_bucket_tensors_of_same_lengths() -> None:
    dataloader = (
        fairseq2.data.read_sequence(range(9))
        .map(lambda l: torch.ones(5))
        .bucket(4)
        .map(Collater())
        .map(lambda x: x.shape)
        .and_return()
    )
    assert_eq_twice(dataloader, [[4, 5], [4, 5], [1, 5]])


def test_bucket_tensors_of_different_lengths() -> None:
    raw_lengths = [1, 5, 6, 1, 1, 1, 7, 1, 1]
    dataloader = (
        fairseq2.data.read_sequence(raw_lengths)
        .map(lambda l: torch.ones(l) * l)
        .bucket(4)
        .map(Collater(pad_idx=0))
        .map(lambda x: x.shape)
        .and_return()
    )
    assert_eq_twice(dataloader, [[4, 6], [4, 7], [1, 1]])

    # Without padding we can't batch tensors of different lengths
    dataloader = (
        fairseq2.data.read_sequence(raw_lengths)
        .map(lambda l: torch.ones(l) * l)
        .bucket(4)
        .map(Collater())
        .map(lambda x: x.shape)
        .and_return()
    )
    with pytest.raises(
        RuntimeError, match="stack expects each tensor to be equal size"
    ):
        list(dataloader)


def test_bucket_tuples() -> None:
    dataloader = (
        fairseq2.data.text.read_text(FILE, rtrim=True)
        # Batching tuples yields tuples of lists.
        .map(lambda l: (l, len(l)))
        .bucket(8)
        .map(Collater())
        .and_return()
    )
    dataset = list(dataloader)

    assert [l for (lines, lengths) in dataset for l in lines] == FILE_LINES

    N = len(FILE_LINES)
    expected_batch_sizes = [8] * (N // 8)
    if N % 8 != 0:
        expected_batch_sizes.append(N % 8)

    assert [len(lines) for lines, lengths in dataset] == expected_batch_sizes
    assert [len(lengths) for lines, lengths in dataset] == expected_batch_sizes
    for lines, lengths in dataset:
        assert lengths == [len(line) for line in lines]


def test_bucket_by_length() -> None:
    raw_lengths = [1, 5, 6, 1, 1, 1, 7, 1, 1]
    dataloader = (
        fairseq2.data.read_sequence(raw_lengths)
        .map(lambda l: torch.ones(l) * l)
        # We want batches of 4 one-length elements
        # and batches of 3 seven-length elements
        .bucket_by_length([(4, 1), (3, 7)], max_data_length=7)
        .map(Collater(pad_idx=0))
        .map(lambda x: x.tolist())
        .and_return()
    )
    assert_eq_twice(
        dataloader,
        [
            [[1], [1], [1], [1]],
            [[5, 5, 5, 5, 5, 0, 0], [6, 6, 6, 6, 6, 6, 0], [7, 7, 7, 7, 7, 7, 7]],
            [[1], [1]],
        ],
    )


def test_bucket_by_length_2D() -> None:
    raw_lengths = [1, 5, 6, 1, 1, 1, 7, 1, 1]
    dataloader = (
        fairseq2.data.read_sequence(raw_lengths)
        .map(lambda l: torch.ones(l, 2) * l)
        .bucket_by_length([(4, 1), (3, 7)], max_data_length=7)
        .map(Collater(pad_idx=0))
        .map(lambda t: t.shape)
        .and_return()
    )

    assert_eq_twice(dataloader, [[4, 1, 2], [3, 7, 2], [2, 1, 2]])


@pytest.mark.xfail(reason="https://github.com/fairinternal/fairseq2/issues/267")
def test_bucket_by_length_can_resume(tmp_path: Path) -> None:
    raw_lengths = [1, 5, 6, 1, 1, 1, 7, 1, 1]
    dataloader = (
        fairseq2.data.read_sequence(raw_lengths)
        .map(lambda l: torch.ones(l) * l)
        # We want batches of 4 one-length elements
        # and batches of 3 seven-length elements
        .bucket_by_length([(4, 1), (3, 7)], max_data_length=7)
        .map(Collater(pad_idx=0))
        .and_return()
    )
    it = iter(dataloader)
    # Read first batch
    next(it)

    torch.save(dataloader.state_dict(), tmp_path / "dataloader.pt")
    # exhaust the full iterator
    rest_of_data = [t.shape for t in it]
    assert rest_of_data == [(3, 7), (2, 1)]

    # Restore the state of dataloader to after first batch
    dataloader.load_state_dict(torch.load(tmp_path / "dataloader.pt"))
    assert rest_of_data == [t.shape for t in it]


def assert_eq_twice(
    dataloader: fairseq2.data.DataPipeline, expected: List[Any]
) -> None:
    # First epoch
    assert list(dataloader) == expected

    dataloader.reset()

    # Second epoch
    assert list(dataloader) == expected
