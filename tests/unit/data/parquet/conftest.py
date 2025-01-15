import shutil
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def get_random_table(size: int, seed: int = 123) -> pa.Table:
    rds = np.random.RandomState(seed)
    data = {
        "cat": rds.randint(0, 10, size),
        "name": ["name_" + str(i) for i in range(size)],
        "score": np.round(rds.randn(size), 7),
    }
    return pa.Table.from_pydict(data)


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
