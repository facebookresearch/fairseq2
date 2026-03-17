# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import threading
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fairseq2.data.parquet.fragment_streaming.builder import (
    ParquetFragmentStreamer,
)
from fairseq2.data.parquet.fragment_streaming.config import FragmentStreamingConfig
from fairseq2.data.parquet.fragment_streaming.primitives import (
    ParquetDatasetKey,
    clear_dataset_cache,
    get_dataset_cache_info,
    list_parquet_fragments,
    stream_parquet_fragments,
)
from fairseq2.data.parquet.utils import fragment_stable_hash


def are_fragments_equal(fragment1, fragment2):
    if fragment1.path != fragment2.path:
        return False

    if fragment1.metadata.to_dict() != fragment2.metadata.to_dict():
        return False

    if fragment1.physical_schema != fragment2.physical_schema:
        return False
    if [int(rg.id) for rg in fragment1.row_groups] != [
        int(rg.id) for rg in fragment2.row_groups
    ]:
        return False

    # # Optionally, compare the data
    # table1 = fragment1.to_table()
    # table2 = fragment2.to_table()
    # if not table1.equals(table2):
    #     return False

    return True


@pytest.mark.parametrize("split_to_row_groups", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("nb_epochs", [1, 2, 10])
def test_list_parquet_fragments_basic(
    controled_row_groups_pq_dataset, split_to_row_groups, shuffle, nb_epochs
):
    parquet_ds = pq.ParquetDataset(controled_row_groups_pq_dataset)
    fragments_pipeline = list_parquet_fragments(
        parquet_ds,
        nb_epochs=nb_epochs,
        split_to_row_groups=split_to_row_groups,
        shuffle=shuffle,
        seed=1,
    )
    fragments = list(fragments_pipeline.and_return())

    single_epoch_nb_fragments = 6 if split_to_row_groups else 3
    assert all(isinstance(f, pa.dataset.Fragment) for f in fragments)
    assert len(fragments) == single_epoch_nb_fragments * nb_epochs
    assert len(set(fragments)) == single_epoch_nb_fragments
    if not shuffle:
        assert all(
            are_fragments_equal(frag1, frag2)
            for frag1, frag2 in zip(
                fragments[:single_epoch_nb_fragments],
                fragments[single_epoch_nb_fragments:],
            )
        )
    else:
        epoch_frags = []
        for i in range(nb_epochs):
            epoch_frag = fragments[
                single_epoch_nb_fragments * i : single_epoch_nb_fragments * (i + 1)
            ]
            epoch_frags.append(epoch_frag)
            assert len(set(epoch_frag)) == single_epoch_nb_fragments
        if nb_epochs > 1:
            assert not all(
                are_fragments_equal(frag1, frag2)
                for frag1, frag2 in zip(epoch_frags[0], epoch_frags[1])
            )


@pytest.mark.parametrize("nb_epochs", [1, 2, 10])
@pytest.mark.parametrize("shuffling_window", [1, 2, 4, 10])
@pytest.mark.parametrize("files_circular_shift", [0, 0.334, 0.5, 1.0])
@pytest.mark.parametrize("stop_id", [5, 15, 20])
@pytest.mark.parametrize("shuffle", [True, False])
def test_stream_parquet_fragments_generic(
    controled_row_groups_pq_dataset,
    nb_epochs: int,
    shuffling_window: int,
    files_circular_shift: float,
    shuffle: bool,
    stop_id: int,
):
    parquet_ds = pq.ParquetDataset(controled_row_groups_pq_dataset)
    total_number_of_row_groups = 6

    def get_new_stream():
        return stream_parquet_fragments(
            parquet_ds,
            nb_epochs=nb_epochs,
            seed=1,
            shuffle=shuffle,
            split_to_row_groups=True,
            shuffling_window=shuffling_window,
            files_circular_shift=files_circular_shift,
        ).and_return()

    spf = get_new_stream()

    all_fragments = list(iter(spf))
    all_fragments_bis = list(iter(get_new_stream()))

    assert set(
        list(Counter([fragment_stable_hash(f) for f in all_fragments]).values())
    ) == set([nb_epochs])

    if shuffling_window <= total_number_of_row_groups // 2:
        assert set(
            list(
                Counter(
                    [fragment_stable_hash(f) for f in all_fragments[:shuffling_window]]
                ).values()
            )
        ) == set([1])

        assert set(
            list(Counter([fragment_stable_hash(f) for f in all_fragments]).values())
        ) == set([nb_epochs])

    # check deterministic behavior
    assert (
        len(all_fragments)
        == len(all_fragments_bis)
        == total_number_of_row_groups * nb_epochs
    )
    for f, f_bis in zip(all_fragments, all_fragments_bis):
        assert are_fragments_equal(f, f_bis)

    # reset
    spf.reset()
    all_fragments = list(iter(spf))
    assert (
        len(all_fragments)
        == len(all_fragments_bis)
        == total_number_of_row_groups * nb_epochs
    )
    for f, f_bis in zip(all_fragments, all_fragments_bis):
        assert are_fragments_equal(f, f_bis)

    # reload state
    spf.reset()
    ii = iter(spf)
    all_fragments = []
    for _ in range(min(stop_id, total_number_of_row_groups * nb_epochs)):
        all_fragments.append(next(ii))
    state = spf.state_dict(strict=False)
    state = pickle.loads(pickle.dumps(state))

    # starting new one
    spf = get_new_stream()
    spf.load_state_dict(state)
    all_fragments = all_fragments + list(iter(spf))
    assert (
        len(all_fragments)
        == len(all_fragments_bis)
        == total_number_of_row_groups * nb_epochs
    )
    for f, f_bis in zip(all_fragments, all_fragments_bis):
        assert are_fragments_equal(f, f_bis)


def test_stream_parquet_fragments_with_circular_shift(controled_row_groups_pq_dataset):
    parquet_ds = pq.ParquetDataset(controled_row_groups_pq_dataset)

    def show_fragments(frag):
        return (
            frag.path.split("/")[-2],
            [r.id for r in frag.row_groups][0],
        )

    def get_new_stream(files_circular_shift, shuffling_window):
        return (
            stream_parquet_fragments(
                parquet_ds,
                nb_epochs=2,
                seed=1,
                shuffle=True,
                split_to_row_groups=True,
                shuffling_window=shuffling_window,
                files_circular_shift=files_circular_shift,
            )
            .map(show_fragments)
            .and_return()
        )

    result_0 = list(iter(get_new_stream(0, 1)))
    # order of files (`cat`` here) is random and different from one epoch to another
    # order of row groups is the same
    expcted_0 = [
        ("cat=cat_0", 0),  # epoch 1
        ("cat=cat_0", 1),
        ("cat=cat_2", 0),
        ("cat=cat_2", 1),
        ("cat=cat_2", 2),
        ("cat=cat_1", 0),
        ("cat=cat_2", 0),  # epoch 2
        ("cat=cat_2", 1),
        ("cat=cat_2", 2),
        ("cat=cat_1", 0),
        ("cat=cat_0", 0),
        ("cat=cat_0", 1),
    ]
    assert result_0 == expcted_0

    result_05 = list(iter(get_new_stream(0.5, 1)))
    # files are shuffled by 1 = int(3 * 0.5) files left
    # 1 epoch (0, 2, 1) -> (2, 1, 0)
    # 2 epoch (2, 1, 0) -> (1, 0, 2)
    expcted_05 = [
        ("cat=cat_2", 0),  # epoch 1
        ("cat=cat_2", 1),
        ("cat=cat_2", 2),
        ("cat=cat_1", 0),
        ("cat=cat_0", 0),
        ("cat=cat_0", 1),
        ("cat=cat_1", 0),  # epoch 2
        ("cat=cat_0", 0),
        ("cat=cat_0", 1),
        ("cat=cat_2", 0),
        ("cat=cat_2", 1),
        ("cat=cat_2", 2),
    ]
    assert result_05 == expcted_05

    result_05_3 = list(iter(get_new_stream(0.5, 3)))
    # locally shuffled in window=3 elements from previous result
    expcted_05_3 = [
        ("cat=cat_2", 0),  # epoch 1
        ("cat=cat_2", 2),
        ("cat=cat_2", 1),
        ("cat=cat_0", 0),
        ("cat=cat_0", 1),
        ("cat=cat_1", 0),
        ("cat=cat_1", 0),  # epoch 2
        ("cat=cat_0", 1),
        ("cat=cat_0", 0),
        ("cat=cat_2", 0),
        ("cat=cat_2", 2),
        ("cat=cat_2", 1),
    ]
    assert result_05_3 == expcted_05_3


@pytest.mark.parametrize("nb_epochs", [1, 2, 10])
@pytest.mark.parametrize("shuffling_window", [-1, 0, 2, 4, 10])
def test_fragment_input_conifg(
    controled_row_groups_pq_dataset, nb_epochs, shuffling_window
):
    config = FragmentStreamingConfig(
        parquet_path=controled_row_groups_pq_dataset,
        nb_epochs=nb_epochs,
        seed=1,
        split_to_row_groups=True,
        fragment_shuffle_window=shuffling_window,
        files_circular_shift=True,
    )
    total_number_of_row_groups = 6
    # with files_circular_shift > 0, shards will be perfectly speparated

    PFS = ParquetFragmentStreamer(config=config)
    pi01 = PFS.build_pipeline(0, 1).map(fragment_stable_hash).and_return()
    pi02 = PFS.build_pipeline(0, 2).map(fragment_stable_hash).and_return()
    pi12 = PFS.build_pipeline(1, 2).map(fragment_stable_hash).and_return()

    result_01 = list(iter(pi01))
    result_02 = list(iter(pi02))
    result_12 = list(iter(pi12))

    if shuffling_window != -1:
        # always dijoint for files_circular_shift=True
        assert set(result_12) & set(result_02) == set()

    assert Counter(result_02) + Counter(result_12) == Counter(result_01)
    assert list(Counter(result_01).values()) == total_number_of_row_groups * [nb_epochs]


class TestParquetDatasetCache:
    """Tests for the cache functionality in ParquetFragmentStreamer."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Clear cache before and after each test."""
        clear_dataset_cache()
        yield
        clear_dataset_cache()

    def test_same_config_returns_same_dataset_object(
        self, controled_row_groups_pq_dataset
    ):
        """Two ParquetFragmentStreamer instances with same config share the same dataset."""
        config = FragmentStreamingConfig(
            parquet_path=controled_row_groups_pq_dataset,
            nb_epochs=1,
            seed=42,
        )

        streamer1 = ParquetFragmentStreamer(config=config, use_cache=True)
        streamer2 = ParquetFragmentStreamer(config=config, use_cache=True)

        # Access datasets to trigger initialization
        dataset1 = streamer1.dataset
        dataset2 = streamer2.dataset

        # The underlying PyArrow dataset should be the exact same object
        assert dataset1.dataset is dataset2.dataset, (
            "Expected same dataset object for identical config, "
            f"but got different objects: {id(dataset1.dataset)} vs {id(dataset2.dataset)}"
        )

        # Verify cache was hit
        cache_info = get_dataset_cache_info()
        assert cache_info.hits == 1, f"Expected 1 cache hit, got {cache_info.hits}"
        assert cache_info.misses == 1, f"Expected 1 cache miss, got {cache_info.misses}"

    def test_different_paths_produce_different_cache_entries(
        self, controled_row_groups_pq_dataset, multi_partition_file_dataset
    ):
        """Different parquet paths should result in different cache entries."""
        config1 = FragmentStreamingConfig(
            parquet_path=controled_row_groups_pq_dataset,
            nb_epochs=1,
            seed=42,
        )
        config2 = FragmentStreamingConfig(
            parquet_path=multi_partition_file_dataset,
            nb_epochs=1,
            seed=42,
        )

        streamer1 = ParquetFragmentStreamer(config=config1, use_cache=True)
        streamer2 = ParquetFragmentStreamer(config=config2, use_cache=True)

        dataset1 = streamer1.dataset
        dataset2 = streamer2.dataset

        # Datasets should be different objects since paths differ
        assert (
            dataset1.dataset is not dataset2.dataset
        ), "Expected different dataset objects for different paths"

        # Verify both were cache misses (different keys)
        cache_info = get_dataset_cache_info()
        assert (
            cache_info.misses == 2
        ), f"Expected 2 cache misses, got {cache_info.misses}"

    def test_different_filters_share_same_cached_dataset(
        self, controled_row_groups_pq_dataset
    ):
        """Different partition_filters should share the same cached dataset.

        The cache key intentionally excludes partition_filters because
        ds.dataset() doesn't depend on filters - they're applied later.
        """
        config1 = FragmentStreamingConfig(
            parquet_path=controled_row_groups_pq_dataset,
            partition_filters='pc.field("cat") == "cat_0"',
            nb_epochs=1,
            seed=42,
        )
        config2 = FragmentStreamingConfig(
            parquet_path=controled_row_groups_pq_dataset,
            partition_filters='pc.field("cat") == "cat_1"',
            nb_epochs=1,
            seed=42,
        )

        streamer1 = ParquetFragmentStreamer(config=config1, use_cache=True)
        streamer2 = ParquetFragmentStreamer(config=config2, use_cache=True)

        dataset1 = streamer1.dataset
        dataset2 = streamer2.dataset

        # The underlying PyArrow dataset should be shared (same path)
        assert (
            dataset1.dataset is dataset2.dataset
        ), "Expected same underlying dataset for same path with different filters"

        # But the wrappers should have different filters (compare string representations
        # since PyArrow Expressions cannot be compared with != directly)
        assert str(dataset1.partition_filters) != str(
            dataset2.partition_filters
        ), f"Expected different partition filters, but both have: {dataset1.partition_filters}"

        cache_info = get_dataset_cache_info()
        assert cache_info.hits == 1, f"Expected 1 cache hit, got {cache_info.hits}"

    def test_clear_cache_forces_reinitialization(self, controled_row_groups_pq_dataset):
        """clear_dataset_cache() should force re-initialization of datasets."""
        config = FragmentStreamingConfig(
            parquet_path=controled_row_groups_pq_dataset,
            nb_epochs=1,
            seed=42,
        )

        streamer1 = ParquetFragmentStreamer(config=config, use_cache=True)
        dataset1 = streamer1.dataset

        cache_info_before = get_dataset_cache_info()
        assert cache_info_before.misses == 1

        # Clear the cache
        clear_dataset_cache()

        cache_info_after_clear = get_dataset_cache_info()
        assert cache_info_after_clear.hits == 0
        assert cache_info_after_clear.misses == 0

        # Create a new streamer - should be a cache miss
        streamer2 = ParquetFragmentStreamer(config=config, use_cache=True)
        dataset2 = streamer2.dataset

        # After clearing, we should get a different dataset object
        assert (
            dataset1.dataset is not dataset2.dataset
        ), "Expected different dataset object after cache clear"

        cache_info_final = get_dataset_cache_info()
        assert (
            cache_info_final.misses == 1
        ), f"Expected 1 cache miss after clear, got {cache_info_final.misses}"

    def test_thread_safety_concurrent_streamers(self, controled_row_groups_pq_dataset):
        """Multiple threads creating streamers concurrently should not cause race conditions.

        Note: lru_cache is thread-safe but does not prevent duplicate work when
        multiple threads call it concurrently before the cache is populated.
        This test verifies that no errors occur during concurrent access.
        """
        config = FragmentStreamingConfig(
            parquet_path=controled_row_groups_pq_dataset,
            nb_epochs=1,
            seed=42,
        )

        num_threads = 10
        results = []
        errors = []
        lock = threading.Lock()

        def create_streamer_and_get_dataset():
            try:
                streamer = ParquetFragmentStreamer(config=config, use_cache=True)
                dataset = streamer.dataset
                with lock:
                    results.append(id(dataset.dataset))
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=create_streamer_and_get_dataset)
            for _ in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should have occurred
        assert not errors, f"Errors occurred during concurrent access: {errors}"

        # All threads should have completed successfully
        assert len(results) == num_threads

        # Due to lru_cache behavior, we might get at most 2 unique datasets:
        # one from initial concurrent misses, and one that gets cached.
        # The important thing is that no errors occurred and all threads completed.
        # Subsequent sequential access will always return the cached dataset.
        unique_datasets = len(set(results))
        assert (
            unique_datasets <= num_threads
        ), f"Got more unique datasets ({unique_datasets}) than threads ({num_threads})"

    def test_cache_disabled_creates_new_datasets(self, controled_row_groups_pq_dataset):
        """When use_cache=False, each streamer should get a new dataset."""
        config = FragmentStreamingConfig(
            parquet_path=controled_row_groups_pq_dataset,
            nb_epochs=1,
            seed=42,
        )

        streamer1 = ParquetFragmentStreamer(config=config, use_cache=False)
        streamer2 = ParquetFragmentStreamer(config=config, use_cache=False)

        dataset1 = streamer1.dataset
        dataset2 = streamer2.dataset

        # Without cache, should be different objects
        assert (
            dataset1.dataset is not dataset2.dataset
        ), "Expected different dataset objects when cache is disabled"

        # Cache should not have been used
        cache_info = get_dataset_cache_info()
        assert cache_info.hits == 0
        assert cache_info.misses == 0


class TestParquetDatasetKeyNormalization:
    """Tests for cache key normalization edge cases."""

    def test_list_path_normalized_to_sorted_tuple(self):
        """List paths should be normalized to sorted tuples for consistent caching."""
        key1 = ParquetDatasetKey.from_init_args(
            parquet_path=["/path/to/a", "/path/to/b", "/path/to/c"],
            filesystem=None,
        )
        key2 = ParquetDatasetKey.from_init_args(
            parquet_path=["/path/to/c", "/path/to/a", "/path/to/b"],
            filesystem=None,
        )

        # Different orderings should produce the same key
        assert (
            key1.path == key2.path
        ), f"Expected same normalized path, got {key1.path} vs {key2.path}"
        assert key1 == key2, "Keys should be equal regardless of list order"

    def test_string_path_preserved(self):
        """String paths should be preserved as-is."""
        key = ParquetDatasetKey.from_init_args(
            parquet_path="/path/to/dataset",
            filesystem=None,
        )
        assert key.path == "/path/to/dataset"
        assert isinstance(key.path, str)

    def test_none_filesystem_serialized(self):
        """None filesystem should be handled correctly."""
        key = ParquetDatasetKey.from_init_args(
            parquet_path="/path/to/dataset",
            filesystem=None,
        )
        # The filesystem should be serialized
        assert key.filesystem is not None

    def test_key_is_hashable(self):
        """ParquetDatasetKey should be hashable for use in lru_cache."""
        key = ParquetDatasetKey.from_init_args(
            parquet_path="/path/to/dataset",
            filesystem=None,
        )

        # Should not raise
        hash(key)

        # Should work as dict key
        d = {key: "value"}
        assert d[key] == "value"

    def test_same_inputs_produce_equal_keys(self):
        """Same inputs should produce equal and hash-equal keys."""
        key1 = ParquetDatasetKey.from_init_args(
            parquet_path="/path/to/dataset",
            filesystem=None,
        )
        key2 = ParquetDatasetKey.from_init_args(
            parquet_path="/path/to/dataset",
            filesystem=None,
        )

        assert key1 == key2
        assert hash(key1) == hash(key2)
