# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from pathlib import Path
from typing import Any, Iterator, List, Optional

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


def test_state_dict(tmp_path: Path) -> None:
    dataloader = fairseq2.data.text.read_text(FILE, rtrim=True).and_return()
    it = iter(dataloader)
    # Read first line
    next(it)
    assert_checkpoint_works(dataloader, it, tmp_path / "dataloader.pt")


def test_batch() -> None:
    dataloader = fairseq2.data.text.read_text(FILE, rtrim=True).batch(8).and_return()
    dataloader_drop_remainder = (
        fairseq2.data.text.read_text(FILE, rtrim=True)
        .batch(8, drop_remainder=True)
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


def test_batch_tensors_of_same_lengths() -> None:
    dataloader = (
        fairseq2.data.read_sequence(range(9))
        .map(lambda l: torch.ones(5))
        .batch(4)
        .map(lambda x: x.shape)
        .and_return()
    )
    assert_eq_twice(dataloader, [[4, 5], [4, 5], [1, 5]])


def test_batch_tensors_of_different_lengths() -> None:
    raw_lengths = [1, 5, 6, 1, 1, 1, 7, 1, 1]
    dataloader = (
        fairseq2.data.read_sequence(raw_lengths)
        .map(lambda l: torch.ones(l) * l)
        .batch(4, pad_idx=0)
        .map(lambda x: x.shape)
        .and_return()
    )
    assert_eq_twice(dataloader, [[4, 6], [4, 7], [1, 1]])

    # Without padding we can't batch tensors of different lengths
    dataloader = (
        fairseq2.data.read_sequence(raw_lengths)
        .map(lambda l: torch.ones(l) * l)
        .batch(4)
        .map(lambda x: x.shape)
        .and_return()
    )
    with pytest.raises(
        RuntimeError, match="stack expects each tensor to be equal size"
    ):
        list(dataloader)


def test_batch_tuples() -> None:
    dataloader = (
        fairseq2.data.text.read_text(FILE, rtrim=True)
        # Batching tuples yields tuples of lists.
        .map(lambda l: (l, len(l)))
        .batch(8)
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


def test_islice() -> None:
    X = list(range(10))

    def assert_islice(start: int, stop: Optional[int], step: Optional[int]) -> None:
        dataloader = (
            fairseq2.data.read_sequence(X).islice(start, stop, step).and_return()
        )
        assert_eq_twice(dataloader, list(itertools.islice(X, start, stop, step)))

    assert_islice(0, 5, 1)
    assert_islice(2, 5, 1)
    assert_islice(5, 5, 1)
    assert_islice(2, 8, 1)
    assert_islice(2, 8, 2)
    assert_islice(2, 8, 3)
    assert_islice(5, 9, 3)


def test_islice_default_step() -> None:
    X = list(range(10))

    def assert_islice(start: int, stop: Optional[int]) -> None:
        dataloader = fairseq2.data.read_sequence(X).islice(start, stop).and_return()
        assert_eq_twice(dataloader, list(itertools.islice(X, start, stop)))

    assert_islice(0, 5)
    assert_islice(3, 5)


def test_islice_stop_signature() -> None:
    X = list(range(10))

    def assert_islice(stop: int) -> None:
        dataloader = fairseq2.data.read_sequence(X).islice(stop).and_return()
        assert_eq_twice(dataloader, list(itertools.islice(X, stop)))

    assert_islice(7)
    assert_islice(30)


def test_islice_next() -> None:
    X = list(range(10))
    dataloader = fairseq2.data.read_sequence(X).islice(2, 8, 3).and_return()
    it = iter(dataloader)
    assert next(it) == 2
    assert next(it) == 5
    with pytest.raises(StopIteration):
        next(it)


def test_islice_edge_cases() -> None:
    X = list(range(10))
    dl_big_start = fairseq2.data.read_sequence(X).islice(15, 20, 4).and_return()
    it = iter(dl_big_start)
    with pytest.raises(StopIteration):
        next(it)

    dl_none_stop = fairseq2.data.read_sequence(X).islice(5, None, 3).and_return()
    it = iter(dl_none_stop)
    assert 5 == next(it)
    assert 8 == next(it)
    with pytest.raises(StopIteration):
        next(it)


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
        list(dataloader)


def test_filter() -> None:
    X = list(range(10))
    Y = [x for x in X if x % 2 == 0]

    dataloader = (
        fairseq2.data.read_sequence(X).filter(lambda x: x % 2 == 0).and_return()
    )
    assert_eq_twice(dataloader, Y)


def test_shuffle() -> None:
    X = list(range(100))
    dataloader = fairseq2.data.read_sequence(X).shuffle(16, seed=42).and_return()
    data = list(dataloader)
    assert sorted(data) == X
    # Because of our buffer we are biased, first elements, will tend to appear first.
    assert max(data[:10]) < 16 + 10

    # fmt: off
    assert data ==  [
        3,8,7,11,13,16,12,4,14,21,22,1,24,25,26,27,18,2,28,35,36,37,38,39,40,41,9,32,44,45,46,47,48,17,50,51,52,53,20,23,56,57,58,10,34,55,62,33,64,60,49,65,63,31,15,71,54,0,74,42,19,77,78,79,69,81,70,5,84,29,86,87,75,88,61,89,85,83,94,76,90,97,98,73,82,91,92,30,80,6,67,96,43,99,59,93,68,95,66,72
    ]
    # fmt: on

    # For the second epoch the order should be different.
    snd_epoch = list(dataloader)
    assert sorted(snd_epoch) == X
    assert snd_epoch != data


def test_reproducible_shuffle() -> None:
    """
    Makes sure that two independent shuffles with same seed will yield in the same order.
    """
    X = list(range(100))
    d1 = (
        fairseq2.data.read_sequence(X)
        .map(lambda x: x + 1)
        .shuffle(16, seed=54)
        .and_return()
    )
    d2 = (
        fairseq2.data.read_sequence(X)
        .shuffle(16, seed=54)
        .map(lambda x: x + 1)
        .and_return()
    )

    assert list(d1) == list(d2)


def test_deterministic_shuffle(tmp_path: Path) -> None:
    X = list(range(100))
    dataloader = (
        fairseq2.data.read_sequence(X).shuffle(16, deterministic=True).and_return()
    )

    it = iter(dataloader)
    [next(it) for _ in range(20)]
    assert_checkpoint_works(dataloader, it, tmp_path / "dataloader.pt")


def assert_eq_twice(
    dataloader: fairseq2.data.DataPipeline, expected: List[Any]
) -> None:
    # First epoch
    assert list(dataloader) == expected
    # Second epoch
    assert list(dataloader) == expected


def assert_checkpoint_works(
    dataloader: fairseq2.data.DataPipeline, it: Iterator[Any], tmp_path: Path
) -> None:
    torch.save(dataloader.state_dict(), tmp_path)
    # exhaust the full iterator
    rest_of_data = list(it)

    # Restore the state of dataloader to after first line
    dataloader.load_state_dict(torch.load(tmp_path))
    assert rest_of_data == list(it)
