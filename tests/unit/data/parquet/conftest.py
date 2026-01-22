# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq


def get_random_table(size: int, seed: int = 123) -> pa.Table:
    rds = np.random.RandomState(seed)
    data = {
        "cat": rds.randint(0, 10, size),
        "name": ["name_" + str(i) for i in range(size)],
        "score": np.round(rds.randn(size), 7),
    }
    return pa.Table.from_pydict(data)


def create_sample_parquet_dataset(
    base_dir: str,
    num_files: int = 3,
    num_row_groups: int = 2,
    partition_key: Optional[str] = None,
    partition_value: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> List[pa.Table]:
    """
    Create a sample Parquet dataset with optional partitioning.

    This function generates multiple Parquet files (by default, 3),
    each containing a small PyArrow table. The number of row groups
    is also configurable. Optionally, a partition column (key/value)
    can be added to each table to simulate partitioning scenarios.
    A random seed may be specified for reproducible random data.

    Args:
        base_dir (str): Directory in which to create the Parquet files.
        num_files (int, optional): Number of Parquet files to generate.
            Defaults to 3.
        num_row_groups (int, optional): Number of row groups per file.
            Defaults to 2.
        partition_key (str, optional): Column name for partitioning (if any).
            Defaults to None.
        partition_value (str, optional): Partition value for the added partition key.
            Defaults to None.
        random_seed (int, optional): Seed for NumPy's random generator.
            If not provided, data will be randomized each time.

    Returns:
        List[pa.Table]: List of PyArrow tables that were written to disk.

    Raises:
        ValueError: If the number of row groups is zero or negative.

    Example:
        >>> create_sample_parquet_dataset("sample_data", num_files=2, num_row_groups=1)
        [pyarrow.Table, pyarrow.Table]
    """
    if num_row_groups <= 0:
        raise ValueError("num_row_groups must be a positive integer.")

    if random_seed is not None:
        np.random.seed(random_seed)

    os.makedirs(base_dir, exist_ok=True)
    tables = []

    # Decide how many rows go into each row group.
    # For simplicity, assume total rows is 10, then divide by num_row_groups.
    # Adjust as needed for your use case.
    row_group_size = np.ceil(10 // num_row_groups)

    for i in range(num_files):
        # Create a simple dataset of 'id' and 'value'
        start_idx = i * 10
        end_idx = (i + 1) * 10
        data = {
            "id": np.arange(start_idx, end_idx),
            "value": np.random.rand(10),
        }

        table = pa.table(data)

        # Add partition column if both partition_key and partition_value are provided
        if partition_key and partition_value:
            partition_array = pa.array([partition_value] * 10)
            table = table.append_column(partition_key, partition_array)  # type: ignore

        # Write the table to a Parquet file
        pq_file_path = os.path.join(base_dir, f"data_{i}.parquet")
        pq.write_table(table, pq_file_path, row_group_size=row_group_size)

        # Collect the in-memory table for return
        tables.append(table)

    return tables


@pytest.fixture()
def multi_partition_file_dataset() -> Generator[Path, None, None]:
    tmpdir = tempfile.mkdtemp()
    tmp_parquet_ds_path = Path(tmpdir) / "test2"

    table = get_random_table(10**3)
    pq.write_to_dataset(table, tmp_parquet_ds_path, partition_cols=["cat"])

    yield tmp_parquet_ds_path
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_table() -> pa.Table:
    return pa.Table.from_pydict(
        {
            "id": [1, 2, 3],
            "tokens": [
                [1, 2, 3],
                [4, 5],
                [6],
            ],
        }
    )


@pytest.fixture
def prefix_array() -> pa.Array:
    return pa.array([0], type=pa.int32())


@pytest.fixture
def suffix_array() -> pa.Array:
    return pa.array([999], type=pa.int32())


@pytest.fixture()
def controled_row_groups_pq_dataset(
    row_groups_size_distribution: List[int] = [2, 1, 3], row_group_size: int = 10
) -> Generator[Path, None, None]:
    total_size = sum(row_groups_size_distribution) * row_group_size

    data = {
        "cat": [
            f"cat_{j}"
            for j, size in enumerate(row_groups_size_distribution)
            for _ in range(size * 10)
        ],
        "id": [f"id_{i}" for i in range(total_size)],
        "seq": [np.arange(i % 10 + 2) for i in range(total_size)],
    }
    table = pa.Table.from_pydict(data)

    tmp_dir = Path(tempfile.gettempdir()) / "parquet_dataset_test"
    tmp_parquet_ds_path = tmp_dir / "test2"

    pq.write_to_dataset(
        table,
        tmp_parquet_ds_path,
        partition_cols=["cat"],
        **{"row_group_size": row_group_size},
    )

    yield tmp_parquet_ds_path
    shutil.rmtree(str(tmp_dir))
