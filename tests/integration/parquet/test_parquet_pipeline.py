import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import torch

from fairseq2.data.parquet.configs import (
    ParquetBasicDataloaderConfig,
    ParquetBatchFormat,
    ParquetDatasetConfig,
    ParquetDatasetLimitOptions,
)
from fairseq2.data.parquet.pipeline import (
    SafeFragment,
    build_iterator_over_one_table,
    list_parquet_fragments,
    parquet_iterator,
)


class TestSafeFragmentIntegration:
    def test_load_all_columns(self, test_parquet_file: SafeFragment):
        """Test loading all columns from a parquet fragment."""
        safe_fragment = SafeFragment(test_parquet_file)
        result = safe_fragment.load()

        # Check that we got all the original columns plus the technical columns
        expected_columns = {
            "col1",
            "col2",
            "col3",
            "__batch_index",
            "__fragment_index",
            "__filename",
            "__index_in_fragement",
            "__row_groups_ids",
        }
        assert set(result.column_names) == expected_columns

        # Verify the data in the columns
        assert result.column("col1").to_pylist() == [1, 2, 3, 4, 5]
        assert result.column("col2").to_pylist() == ["a", "b", "c", "d", "e"]
        assert result.column("col3").to_pylist() == [1.1, 2.2, 3.3, 4.4, 5.5]

    def test_load_specific_columns(self, test_parquet_file: SafeFragment):
        """Test loading specific columns from a parquet fragment."""
        safe_fragment = SafeFragment(test_parquet_file)
        result = safe_fragment.load(columns=["col1", "col3"])

        # Check that we got only the requested columns plus technical columns
        expected_columns = {
            "col1",
            "col3",
            "__batch_index",
            "__fragment_index",
            "__filename",
            "__index_in_fragement",
            "__row_groups_ids",
        }
        assert set(result.column_names) == expected_columns

        # Verify the data in the columns
        assert result.column("col1").to_pylist() == [1, 2, 3, 4, 5]
        assert result.column("col3").to_pylist() == [1.1, 2.2, 3.3, 4.4, 5.5]

    def test_load_nonexistent_columns(self, test_parquet_file):
        """Test loading non-existent columns from a parquet fragment."""
        safe_fragment = SafeFragment(test_parquet_file)
        result = safe_fragment.load(columns=["nonexistent_column"])

        # Should only get technical columns
        expected_columns = {
            "__batch_index",
            "__fragment_index",
            "__filename",
            "__index_in_fragement",
            "__row_groups_ids",
        }
        assert set(result.column_names) == expected_columns

    @pytest.mark.large_file
    def test_load_large_file(self):
        """Test loading a large parquet file (marked as a large file test)."""
        # Create large test data
        large_data = {
            "col1": list(range(100000)),
            "col2": ["data"] * 100000,
            "col3": [1.0] * 100000,
        }
        table = pa.Table.from_pydict(large_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "large_test.parquet"

            # Write the large parquet file with row groups
            pq.write_table(table, file_path, row_group_size=10000)

            # Create dataset and get fragment
            dataset = ds.dataset(file_path, format="parquet")
            fragment = next(dataset.get_fragments())

            safe_fragment = SafeFragment(fragment)
            result = safe_fragment.load()

            # Verify the number of rows
            assert len(result) == 100000

            # Verify some data points
            assert result.column("col1")[0].as_py() == 0
            assert result.column("col1")[-1].as_py() == 99999
            assert result.column("col2")[50000].as_py() == "data"

    def test_fragment_metadata(self, test_parquet_file):
        """Test that fragment metadata is correctly preserved."""
        safe_fragment = SafeFragment(test_parquet_file)
        result = safe_fragment.load()

        # Check that technical columns contain the correct metadata
        filename = result.column("__filename")[0].as_py()
        assert filename.endswith("test.parquet")

        # All rows should have the same fragment index since it's one fragment
        fragment_indices = set(result.column("__fragment_index").to_pylist())
        assert len(fragment_indices) == 1

        # Batch indices should be sequential
        batch_indices = result.column("__batch_index").to_pylist()
        assert batch_indices == [0] * len(result)


class TestSafeFragmentPartitioned:
    def test_load_partitioned_data(self, partitioned_dataset):
        """Test loading data from a partitioned dataset."""
        fragments = list(partitioned_dataset.get_fragments())

        all_data = []
        for fragment in fragments:
            safe_fragment = SafeFragment(fragment)
            result = safe_fragment.load()
            all_data.append(result)

        # Combine all results
        combined = pa.concat_tables(all_data)

        # Verify the total number of rows
        assert len(combined) == 15  # 3 partitions * 5 rows each

        # Verify that we have data from all partitions
        partition_values = set(combined.column("partition").to_pylist())
        assert partition_values == {0, 1, 2}

        # Verify fragment index
        fragment_indices = set(combined.column("__fragment_index").to_pylist())
        assert len(fragment_indices) == 1


class TestListParquetFragments:
    def test_basic_listing(self, multi_row_group_dataset):
        """Test basic fragment listing without filters or splitting."""
        pipeline_builder = list_parquet_fragments(
            multi_row_group_dataset,
            split_to_row_groups=False,
            shuffle_window=None,
        )
        pipeline = pipeline_builder.and_return()

        fragments = list(pipeline)
        assert len(fragments) == 1  # One fragment for the whole file

        # Load and verify the data
        table = fragments[0].to_table()
        assert len(table) == 1000
        assert table.column_names == ["col1", "col2"]
        assert table["col1"][0].as_py() == 0
        assert table["col1"][-1].as_py() == 999

    def test_row_group_splitting(self, multi_row_group_dataset):
        """Test fragment listing with row group splitting."""
        pipeline_builder = list_parquet_fragments(
            multi_row_group_dataset,
            split_to_row_groups=True,
            shuffle_window=None,
        )
        pipeline = pipeline_builder.and_return()

        fragments = list(pipeline)
        assert len(fragments) == 10  # Should have 10 row groups (1000/100)

        # Verify each fragment has correct size
        for fragment in fragments:
            table = fragment.to_table()
            assert len(table) == 100
            assert table.column_names == ["col1", "col2"]

    def test_with_shuffle_window(self, multi_row_group_dataset):
        """Test fragment listing with shuffle window."""
        # Create two pipelines with same seed
        pipeline_builder1 = list_parquet_fragments(
            multi_row_group_dataset,
            split_to_row_groups=True,
            shuffle_window=5,
            seed=42,
        )
        pipeline1 = pipeline_builder1.and_return()

        pipeline_builder2 = list_parquet_fragments(
            multi_row_group_dataset,
            split_to_row_groups=True,
            shuffle_window=5,
            seed=42,
        )
        pipeline2 = pipeline_builder2.and_return()

        # Get fragments from both pipelines
        fragments1 = list(pipeline1)
        fragments2 = list(pipeline2)

        # Convert to tables for comparison
        tables1 = [f.to_table() for f in fragments1]
        tables2 = [f.to_table() for f in fragments2]

        # Check that both pipelines produced same order with same seed
        first_values1 = [t["col1"][0].as_py() for t in tables1]
        first_values2 = [t["col1"][0].as_py() for t in tables2]
        assert first_values1 == first_values2

        # Create pipeline with different seed
        pipeline_builder3 = list_parquet_fragments(
            multi_row_group_dataset,
            split_to_row_groups=True,
            shuffle_window=5,
            seed=43,
        )
        pipeline3 = pipeline_builder3.and_return()

        fragments3 = list(pipeline3)
        tables3 = [f.to_table() for f in fragments3]
        first_values3 = [t["col1"][0].as_py() for t in tables3]

        # Different seed should give different order
        assert first_values1 != first_values3

    def test_with_column_filter(self, multi_row_group_dataset):
        """Test fragment listing with column filtering."""
        pipeline_builder = list_parquet_fragments(
            multi_row_group_dataset,
            limit_options=ParquetDatasetLimitOptions(columns=["col1"]),
            split_to_row_groups=True,
        )
        pipeline = pipeline_builder.and_return()

        fragments = list(pipeline)

        # Currently we are not filtering columns
        for fragment in fragments:
            table = fragment.to_table()
            assert table.column_names == ["col1", "col2"]

    def test_pipeline_reset(self, multi_row_group_dataset):
        """Test that pipeline can be reset and reused."""
        pipeline_builder = list_parquet_fragments(
            multi_row_group_dataset,
            split_to_row_groups=True,
            shuffle_window=None,
        )
        pipeline = pipeline_builder.and_return()

        # First iteration
        fragments1 = list(pipeline)
        tables1 = [f.to_table() for f in fragments1]
        first_values1 = [t["col1"][0].as_py() for t in tables1]

        # Reset and iterate again
        pipeline.reset()
        fragments2 = list(pipeline)
        tables2 = [f.to_table() for f in fragments2]
        first_values2 = [t["col1"][0].as_py() for t in tables2]

        # Should get same order after reset
        assert first_values1 == first_values2


class TestBuildIteratorOverOneTable:
    def test_basic_batching(self, sample_table):
        """Test basic batching without ordering or shuffling."""
        pipeline = build_iterator_over_one_table(
            table=sample_table,
            batch_size=10,
            shuffle=False,
        )

        # Get batches and unwrap _TableWrapper
        wrapped_batches = list(pipeline)
        batches = [batch.table for batch in wrapped_batches]

        assert len(batches) == 10  # 100 rows / 10 batch_size

        # Verify batch sizes
        assert all(len(batch) == 10 for batch in batches)

        # Verify data order is preserved when shuffle=False
        first_values = [batch["int_col"][0].as_py() for batch in batches]
        assert first_values == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    def test_ordered_batching(self, sample_table):
        """Test batching with ordering by length."""
        pipeline = build_iterator_over_one_table(
            table=sample_table,
            batch_size=20,
            order_by_length="list_col",
            shuffle=False,
        )

        wrapped_batches = list(pipeline)
        batches = [batch.table for batch in wrapped_batches]

        # Verify batch sizes
        assert all(len(batch) == 20 for batch in batches)

        # Verify that lengths within each batch are similar
        for batch in batches:
            lengths = [len(lst) for lst in batch["list_col"].to_pylist()]
            max_len_diff = max(lengths) - min(lengths)
            assert max_len_diff <= 1  # Length difference should be minimal

    def test_max_tokens_batching(self, sample_table):
        """Test batching with max_tokens constraint."""
        pipeline = build_iterator_over_one_table(
            table=sample_table,
            max_tokens=50,  # Small max_tokens to force multiple batches
            order_by_length="list_col",
            shuffle=False,
        )

        wrapped_batches = list(pipeline)
        batches = [batch.table for batch in wrapped_batches]

        # Verify that each batch respects max_tokens
        for batch in batches:
            lengths = [len(lst) for lst in batch["list_col"].to_pylist()]
            max_len = max(lengths)
            total_tokens = max_len * len(batch)
            assert total_tokens <= 50

    def test_deterministic_shuffle(self, sample_table):
        """Test that shuffling is deterministic with same seed."""
        # Create two pipelines with same seed
        pipeline1 = build_iterator_over_one_table(
            table=sample_table,
            batch_size=10,
            shuffle=True,
            seed=42,
        )

        pipeline2 = build_iterator_over_one_table(
            table=sample_table,
            batch_size=10,
            shuffle=True,
            seed=42,
        )

        wrapped_batches1 = list(pipeline1)
        wrapped_batches2 = list(pipeline2)

        batches1 = [batch.table for batch in wrapped_batches1]
        batches2 = [batch.table for batch in wrapped_batches2]

        # Compare first values from each batch
        values1 = [batch["int_col"][0].as_py() for batch in batches1]
        values2 = [batch["int_col"][0].as_py() for batch in batches2]

        assert values1 == values2  # Same seed should give same order

        # Create pipeline with different seed
        pipeline3 = build_iterator_over_one_table(
            table=sample_table,
            batch_size=10,
            shuffle=True,
            seed=43,
        )

        wrapped_batches3 = list(pipeline3)
        batches3 = [batch.table for batch in wrapped_batches3]
        values3 = [batch["int_col"][0].as_py() for batch in batches3]

        assert values1 != values3  # Different seeds should give different orders

    def test_parallel_processing(self, sample_table):
        """Test parallel processing with num_parallel_calls."""
        pipeline = build_iterator_over_one_table(
            table=sample_table,
            batch_size=10,
            shuffle=False,
            num_parallel_calls=4,
        )

        wrapped_batches = list(pipeline)
        batches = [batch.table for batch in wrapped_batches]
        assert len(batches) == 10

        # Verify all data is present and correct
        all_values = []
        for batch in batches:
            all_values.extend(batch["int_col"].to_pylist())

        assert sorted(all_values) == list(range(100))

    def test_pipeline_reset(self, sample_table):
        """Test that pipeline can be reset and reused."""
        pipeline = build_iterator_over_one_table(
            table=sample_table,
            batch_size=10,
            shuffle=True,
            seed=42,
        )

        # First iteration
        wrapped_batches1 = list(pipeline)
        batches1 = [batch.table for batch in wrapped_batches1]
        values1 = [batch["int_col"][0].as_py() for batch in batches1]

        # Reset and iterate again
        pipeline.reset()
        wrapped_batches2 = list(pipeline)
        batches2 = [batch.table for batch in wrapped_batches2]
        values2 = [batch["int_col"][0].as_py() for batch in batches2]

        # Should get same order after reset with same seed
        assert values1 == values2

    def test_invalid_config(self, sample_table):
        """Test error handling for invalid configurations."""
        # Test missing both batch_size and max_tokens
        with pytest.raises(ValueError, match="unknown batching method"):
            build_iterator_over_one_table(
                table=sample_table,
                shuffle=False,
            )

        # Test invalid order_by_length column
        with pytest.raises(KeyError):
            pipeline = build_iterator_over_one_table(
                table=sample_table,
                batch_size=10,
                order_by_length="nonexistent_column",
            )
            list(pipeline)  # Force execution


class TestParquetIterator:
    def test_basic_iteration(self, complex_dataset):
        """Test basic iteration with default settings."""
        dataset_config = ParquetDatasetConfig(
            parquet_path=complex_dataset,
            split_to_row_groups=True,
        )

        dataloader_config = ParquetBasicDataloaderConfig(
            batch_size=10, shuffle=False, output_format=ParquetBatchFormat.pyarrow
        )

        batches = list(parquet_iterator(dataset_config, dataloader_config))

        # Should have 30 batches (300 total rows / 10 batch_size)
        assert len(batches) == 30

        # Verify batch size
        assert all(len(batch) == 10 for batch in batches)

        # Verify columns
        assert set(batches[0].column_names) == {"text", "tokens", "length", "partition"}

    def test_torch_output_format(self, complex_dataset):
        """Test iteration with torch output format."""
        dataset_config = ParquetDatasetConfig(
            parquet_path=complex_dataset,
        )

        dataloader_config = ParquetBasicDataloaderConfig(
            batch_size=10, shuffle=False, output_format=ParquetBatchFormat.torch
        )

        batches = list(parquet_iterator(dataset_config, dataloader_config))

        # Verify torch tensor output
        assert isinstance(batches[0]["length"], torch.Tensor)
        assert isinstance(batches[0]["text"], list)
        assert isinstance(batches[0]["tokens"], list)

        # Verify tensor shapes
        assert batches[0]["length"].shape == (10,)

    def test_ordered_batching(self, complex_dataset):
        """Test iteration with length-ordered batching."""
        dataset_config = ParquetDatasetConfig(
            parquet_path=complex_dataset,
        )

        dataloader_config = ParquetBasicDataloaderConfig(
            max_tokens=50, order_by_length="tokens", shuffle=False
        )

        batches = list(parquet_iterator(dataset_config, dataloader_config))

        # Verify max tokens constraint
        for batch in batches:
            lengths = [len(tokens) for tokens in batch["tokens"].to_pylist()]
            max_len = max(lengths)
            assert max_len * len(batch) <= 50

    def test_deterministic_shuffle(self, complex_dataset):
        """Test deterministic shuffling with seeds."""
        dataset_config = ParquetDatasetConfig(
            parquet_path=complex_dataset,
        )

        # Create two configs with same seed
        config1 = ParquetBasicDataloaderConfig(batch_size=10, shuffle=True, seed=42)

        config2 = ParquetBasicDataloaderConfig(batch_size=10, shuffle=True, seed=42)

        batches1 = list(parquet_iterator(dataset_config, config1))
        batches2 = list(parquet_iterator(dataset_config, config2))

        # Verify same order with same seed
        texts1 = [text for batch in batches1 for text in batch["text"].to_pylist()]
        texts2 = [text for batch in batches2 for text in batch["text"].to_pylist()]
        assert texts1 == texts2

        # Different seed should give different order
        config3 = ParquetBasicDataloaderConfig(batch_size=10, shuffle=True, seed=43)

        batches3 = list(parquet_iterator(dataset_config, config3))
        texts3 = [text for batch in batches3 for text in batch["text"].to_pylist()]
        assert texts1 != texts3

    def test_minimum_batch_size(self, complex_dataset):
        """Test minimum batch size filtering."""
        dataset_config = ParquetDatasetConfig(
            parquet_path=complex_dataset,
        )

        dataloader_config = ParquetBasicDataloaderConfig(
            batch_size=10, min_batch_size=10, shuffle=False  # Drop any smaller batches
        )

        batches = list(parquet_iterator(dataset_config, dataloader_config))

        # Verify all batches meet minimum size
        assert all(len(batch) >= 10 for batch in batches)
