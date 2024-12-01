from pathlib import Path
import shutil
import tempfile
from typing import Generator
import pytest
import numpy as np
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

@pytest.fixture()
def multi_partition_file_dataset() -> Generator[Path, None, None]:
    tmpdir = tempfile.mkdtemp()
    tmp_parquet_ds_path = Path(tmpdir) / "test2"

    table = get_random_table(10**3)
    pq.write_to_dataset(table, tmp_parquet_ds_path, partition_cols=["cat"])

    yield tmp_parquet_ds_path
    shutil.rmtree(tmpdir)

