# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fairseq2.data.parquet.pipeline import SafeFragment, init_parquet_dataset
from fairseq2.data.parquet.utils import get_dataset_fragments
from tests.unit.data.parquet.conftest import create_sample_parquet_dataset


@pytest.fixture
def mock_parquet_dataset():
    with patch("fairseq2.data.parquet.pipeline.pq.ParquetDataset") as mock_pd:
        yield mock_pd


@pytest.fixture
def mock_fragment():
    fragment = MagicMock()
    fragment.path = "/path/to/fragment"
    fragment.row_groups = [MagicMock(id=1), MagicMock(id=2)]
    fragment.physical_schema = pa.schema([("col1", pa.int32()), ("col2", pa.string())])
    return fragment


class TestInitParquetDataset:
    def test_init_parquet_dataset_valid_path(
        self, mock_parquet_dataset: MagicMock
    ) -> None:
        parquet_path = "/path/to/parquet"
        filters = None

        # Configure the mock
        mock_instance = MagicMock()
        mock_parquet_dataset.return_value = mock_instance

        # Call the function
        result = init_parquet_dataset(parquet_path, filters)

        # Assertions
        mock_parquet_dataset.assert_called_once_with(
            parquet_path, filters=filters, filesystem=None
        )
        assert result == mock_instance

    def test_init_parquet_dataset_with_filters(
        self, mock_parquet_dataset: MagicMock
    ) -> None:
        parquet_path = "/path/to/parquet"
        filters = pa.dataset.field("column") > 10

        # Configure the mock
        mock_instance = MagicMock()
        mock_parquet_dataset.return_value = mock_instance

        # Call the function
        result = init_parquet_dataset(parquet_path, filters)

        # Assertions
        mock_parquet_dataset.assert_called_once_with(
            parquet_path, filters=filters, filesystem=None
        )
        assert result == mock_instance

    def test_init_parquet_dataset_non_existent_path(
        self, mock_parquet_dataset: MagicMock
    ) -> None:
        parquet_path = "/non/existent/path"
        filters = None

        # Configure the mock to raise an error
        mock_parquet_dataset.side_effect = OSError("File not found")

        # Call the function and assert that the exception is raised
        with pytest.raises(OSError, match="File not found"):
            init_parquet_dataset(parquet_path, filters)

    def test_init_parquet_dataset_empty_dataset(
        self, mock_parquet_dataset: MagicMock
    ) -> None:
        parquet_path = "/path/to/empty/parquet"
        filters = None

        # Configure the mock to return an empty table
        mock_instance = MagicMock()
        mock_instance.schema = pa.schema([])
        mock_parquet_dataset.return_value = mock_instance

        # Call the function
        result = init_parquet_dataset(parquet_path, filters)

        # Assertions
        mock_parquet_dataset.assert_called_once_with(
            parquet_path, filters=filters, filesystem=None
        )
        assert result.schema == pa.schema([])

    def test_init_parquet_dataset_invalid_filters(
        self, mock_parquet_dataset: MagicMock
    ) -> None:
        parquet_path = "/path/to/parquet"
        filters = "invalid_filter_expression"

        # Configure the mock to raise a TypeError
        mock_parquet_dataset.side_effect = TypeError("Invalid filter expression")

        # Call the function and assert that the exception is raised
        with pytest.raises(TypeError, match="Invalid filter expression"):
            init_parquet_dataset(parquet_path, filters)


class TestSafeFragment:
    def test_init(self, mock_fragment: MagicMock) -> None:
        """Test the SafeFragment initialization."""
        safe_fragment = SafeFragment(mock_fragment)
        assert safe_fragment.fragment == mock_fragment

    def test_repr(self, mock_fragment: MagicMock) -> None:
        """Test the SafeFragment representation."""
        safe_fragment = SafeFragment(mock_fragment)
        repr_str = str(safe_fragment)

        assert "SafeFragment" in repr_str
        assert "path = /path/to/fragment" in repr_str
        assert "row_groups = [1, 2]" in repr_str
        assert "physical_schema" in repr_str

    def test_load_without_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a sample dataset with 3 Parquet files
            create_sample_parquet_dataset(tmpdir, num_files=3, num_row_groups=2)

            # Create a ParquetDataset
            dataset = pq.ParquetDataset(tmpdir)

            # Call the function
            fragments = get_dataset_fragments(dataset, None)

            safe_fragment = SafeFragment(fragments[0])
            _ = safe_fragment.load()

            expected_columns = [
                "id",
                "value",
                "__batch_index",
                "__fragment_index",
                "__filename",
            ]
            table = fragments[0].to_table(columns=expected_columns, use_threads=False)
            assert table.column_names == expected_columns
