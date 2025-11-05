import os
import tempfile
from pathlib import Path
from typing import Generator, List

import pytest

pa = pytest.importorskip("pyarrow")

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from fairseq2.data.parquet.fragment_loading.builder import SafeFragment


@pytest.fixture
def test_parquet_file() -> Generator[SafeFragment, None, None]:
    """Create a temporary parquet file with test data."""
    # Create test data
    table = pa.Table.from_pydict(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.1, 2.2, 3.3, 4.4, 5.5],
        }
    )

    # Create a temporary directory and file
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test.parquet"

        # Write the parquet file
        pq.write_table(table, file_path)

        # Create dataset and get fragment
        dataset = ds.dataset(file_path, format="parquet")
        fragment = next(dataset.get_fragments())

        yield fragment


@pytest.fixture
def test_fragments() -> Generator[List[SafeFragment], None, None]:
    """Create multiple parquet fragments for testing."""
    # Create test data for multiple fragments
    tables = []
    for i in range(3):
        table = pa.Table.from_pydict(
            {
                "col1": [i * 10 + j for j in range(5)],
                "col2": [f"frag{i}_{j}" for j in range(5)],
            }
        )
        tables.append(table)

    # Create a temporary directory and files
    with tempfile.TemporaryDirectory() as temp_dir:
        fragments = []
        for i, table in enumerate(tables):
            file_path = Path(temp_dir) / f"test_{i}.parquet"
            pq.write_table(table, file_path)

            # Create dataset and get fragment
            dataset = ds.dataset(file_path, format="parquet")
            fragments.append(next(dataset.get_fragments()))

        yield fragments


@pytest.fixture
def partitioned_dataset() -> Generator[ds.Dataset, None, None]:
    """Create a temporary partitioned dataset for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create partitioned data
        for i in range(3):
            table = pa.Table.from_pydict(
                {
                    "col1": [i * 10 + j for j in range(5)],
                    "col2": [f"part{i}_{j}" for j in range(5)],
                    "partition": [i] * 5,
                }
            )

            partition_dir = Path(temp_dir) / f"partition={i}"
            os.makedirs(partition_dir)

            pq.write_table(
                table.drop(["partition"]),  # Drop partition column as it's in the path
                partition_dir / f"part-{i}.parquet",
            )

        # Create dataset
        dataset = ds.dataset(temp_dir, format="parquet", partitioning="hive")
        yield dataset


@pytest.fixture
def multi_row_group_dataset() -> Generator[str, None, None]:
    """Create a dataset with multiple row groups for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data with enough rows to span multiple row groups
        data = {
            "col1": list(range(1000)),
            "col2": [f"val_{i}" for i in range(1000)],
        }
        table = pa.Table.from_pydict(data)

        file_path = Path(temp_dir) / "multi_group.parquet"
        # Write with small row group size to ensure multiple groups
        pq.write_table(table, file_path, row_group_size=100)

        yield pq.ParquetDataset(str(file_path))


@pytest.fixture
def sample_table() -> pa.Table:
    """Create a sample table with various data types for testing."""
    data = {
        "int_col": list(range(100)),
        "text_col": [f"text_{i}" for i in range(100)],
        "list_col": [[i, i + 1] for i in range(100)],
        "float_col": [float(i) / 2 for i in range(100)],
    }
    return pa.Table.from_pydict(data)


@pytest.fixture
def complex_dataset() -> Generator[str, None, None]:
    """Create a dataset with multiple columns and types for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple parquet files with different partitions
        for partition in range(3):
            data = {
                "text": [f"text_{i}" for i in range(100)],
                "tokens": [[a for a in range(i, i + (i**2 % 10))] for i in range(100)],
                "length": [i**2 % 10 for i in range(100)],
                "partition": [partition] * 100,
            }
            table = pa.Table.from_pydict(data)

            partition_dir = Path(temp_dir) / f"partition={partition}"
            os.makedirs(partition_dir)

            # Write with small row group size
            pq.write_table(
                table.drop(["partition"]),
                partition_dir / f"part-{partition}.parquet",
                row_group_size=20,
            )

        yield str(temp_dir)
