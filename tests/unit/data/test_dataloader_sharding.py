from typing import Any, Callable, Iterator, List

import pytest

import fairseq2.data as dl
from fairseq2.data.data_pipeline import DataPipelineBuilder


@pytest.mark.xfail(reason="shuffle")
def test_shuffle_sharding() -> None:
    f = lambda: dl.read_sequence(range(100)).shuffle(16, seed=54)
    assert_can_shard(f)


def test_islice_sharding() -> None:
    f = lambda: dl.read_sequence(range(20)).islice(2, 17, 2)
    assert_can_shard(f)


def test_map_sharding() -> None:
    f = lambda: dl.read_sequence(range(10)).map(lambda x: x + 10)
    assert_can_shard(f)
    assert_can_shard(f, 3)


def test_filter_sharding() -> None:
    f = lambda: dl.read_sequence(range(20)).filter(lambda x: x % 2 == 0)
    assert_can_shard(f)
    assert_can_shard(f, 3)


def test_round_robin_sharding() -> None:
    dp1 = lambda: dl.read_sequence(range(5)).and_return()
    dp2 = lambda: dl.read_sequence(range(20, 23)).and_return()
    dp3 = lambda: dl.read_sequence(range(100, 107)).and_return()
    f = lambda: dl.round_robin_data_pipelines([dp1(), dp2(), dp3()], [])
    assert_can_shard(f, 3)
    assert_can_shard(f, 5)


def test_zip_sharding() -> None:
    dp1 = lambda: dl.read_sequence(range(10)).and_return()
    dp2 = lambda: dl.read_sequence(range(10, 20)).and_return()

    dp_zip = lambda: dl.zip_data_pipelines([dp1(), dp2()])
    assert_can_shard(dp_zip)


def test_sequence_shrading() -> None:
    f = lambda: dl.read_sequence(range(10))
    assert_can_shard(f)


def assert_can_shard(
    builder_factory: Callable[[], DataPipelineBuilder], n: int = 4
) -> None:
    workers = get_workers(builder_factory, n)
    combined = iterate_workers(workers)
    original_data = list(builder_factory().and_return())

    # not all pipelines handle the last % n elements the same way
    size_to_assert = len(original_data) - (len(original_data) % n)

    assert combined[:size_to_assert] == original_data[:size_to_assert]


def get_workers(
    builder_factory: Callable[[], DataPipelineBuilder], n: int = 4
) -> List[Iterator[Any]]:
    shards = [builder_factory().shard(i, n).and_return() for i in range(n)]
    return [iter(shard) for shard in shards]


def iterate_workers(workers: List[Iterator[Any]]) -> List[Any]:
    done = [False for _ in range(len(workers))]
    combined = []

    while not all(done):
        for i in range(len(workers)):
            try:
                x = next(workers[i])
                combined.append(x)
            except StopIteration:
                done[i] = True

    return combined
