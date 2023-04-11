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


def test_read_text_skip_header() -> None:
    dataloader = fairseq2.data.text.read_text(
        FILE, rtrim=True, skip_header=5
    ).and_return()

    assert_eq_twice(dataloader, FILE_LINES[5:])


def test_state_dict(tmp_path: Path) -> None:
    dataloader = fairseq2.data.text.read_text(FILE, rtrim=True).and_return()
    it = iter(dataloader)
    # Read first line
    next(it)

    torch.save(dataloader.state_dict(), tmp_path / "dataloader.pt")
    # exhaust the full iterator
    rest_of_data = list(it)

    # Restore the state of dataloader to after first line
    dataloader.load_state_dict(torch.load(tmp_path / "dataloader.pt"))
    assert rest_of_data == list(it)


def test_batch() -> None:
    dataloader = (
        fairseq2.data.text.read_text(FILE, rtrim=True).batch(8, False).and_return()
    )
    dataset = list(iter(dataloader))
    assert [l for batch in dataset for l in batch] == FILE_LINES

    N = len(FILE_LINES)
    expected_batch_sizes = [8] * (N // 8)

    if N % 8 != 0:
        expected_batch_sizes.append(N % 8)

    assert [len(batch) for batch in dataset] == expected_batch_sizes


def test_read_sequence() -> None:
    dataloader = fairseq2.data.read_sequence(list(range(10))).and_return()
    assert_eq_twice(dataloader, list(range(10)))


def test_batch_by_length() -> None:
    raw_lengths = [1, 5, 6, 1, 1, 1, 7, 1, 1]
    dataloader = (
        fairseq2.data.read_sequence(raw_lengths)
        .map(lambda l: torch.ones(l) * l)
        # We want batches of 4 one-length elements
        # and batches of 3 seven-length elements
        .batch_by_length([(4, 1), (3, 7)], 0)
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


def test_batch_by_length_2D() -> None:
    raw_lengths = [1, 5, 6, 1, 1, 1, 7, 1, 1]
    dataloader = (
        fairseq2.data.read_sequence(raw_lengths)
        .map(lambda l: torch.ones(l, 2) * l)
        .batch_by_length([(4, 1), (3, 7)], 0)
        .map(lambda t: t.shape)
        .and_return()
    )

    assert_eq_twice(dataloader, [[4, 1, 2], [3, 7, 2], [2, 1, 2]])


@pytest.mark.xfail(reason="https://github.com/fairinternal/fairseq2/issues/267")
def test_batch_by_length_can_resume(tmp_path: Path) -> None:
    raw_lengths = [1, 5, 6, 1, 1, 1, 7, 1, 1]
    dataloader = (
        fairseq2.data.read_sequence(raw_lengths)
        .map(lambda l: torch.ones(l) * l)
        # We want batches of 4 one-length elements
        # and batches of 3 seven-length elements
        .batch_by_length([(4, 1), (3, 7)], 0)
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


def test_batch_by_length_cant_shard() -> None:
    raw_lengths = [1, 4, 5, 4, 1, 4, 6, 4, 1, 4, 1, 4]

    # shard then batch_by_length
    dataloader_shard_then_batch = (
        fairseq2.data.read_sequence(raw_lengths)
        .map(lambda l: torch.ones(l) * l)
        .shard(0, 2)
        .batch_by_length([(4, 1), (3, 7)], 0)
        .map(lambda t: t.shape)
        .and_return()
    )

    assert list(dataloader_shard_then_batch) == [[4, 1], [2, 6]]

    # batch_by_length then shard
    dataloader_batch_then_shard = (
        fairseq2.data.read_sequence(raw_lengths)
        .map(lambda l: torch.ones(l) * l)
        .batch_by_length([(4, 1), (3, 7)], 0)
        .shard(0, 2)
        .map(lambda t: t.shape)
        .and_return()
    )
    with pytest.raises(
        RuntimeError, match="you need to shard before calling 'batched_by_length'"
    ):
        next(iter(dataloader_batch_then_shard))


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 10])
def test_map(chunk_size: int) -> None:
    X = list(range(10))
    X2 = [x**2 for x in X]

    dataloader = (
        fairseq2.data.read_sequence(X)
        .map(lambda x: x**2, chunk_size=chunk_size)
        .and_return()
    )
    assert_eq_twice(dataloader, X2)


def test_map_handle_exceptions() -> None:
    X = list(range(10))

    def contrarian_square(x: int) -> int:
        if x == 4:
            raise RuntimeError("4 is already square enough")
        return x**2

    dataloader = (
        fairseq2.data.read_sequence(X).map(contrarian_square, chunk_size=3).and_return()
    )

    with pytest.raises(RuntimeError, match="4 is already square enough"):
        list(iter(dataloader))


def assert_eq_twice(
    dataloader: fairseq2.data.DataPipeline, expected: List[Any]
) -> None:
    # First epoch
    assert list(iter(dataloader)) == expected
    # Second epoch
    dataloader.reset()
    assert list(iter(dataloader)) == expected
