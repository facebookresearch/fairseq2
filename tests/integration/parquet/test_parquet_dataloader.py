# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import shutil
import string
import tempfile
from collections import Counter
from typing import Any, Dict, Generator, List, Union

import pytest

try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from numpy.typing import NDArray

    from recipes.parquet.parquet_dataloader import (
        ParquetBasicDataloaderConfig,
        ParquetBatchFormat,
        parquet_iterator,
    )
except ImportError:
    pytest.skip("arrow not found", allow_module_level=True)


def gen_random_string(length: int) -> str:
    return "".join(
        random.choice(string.ascii_letters + string.digits) for n in range(length)
    )


def generate_random_pandas_df(size: int, seed: int = 123) -> pd.DataFrame:
    np_rs = np.random.RandomState(seed)
    df: Dict[str, Union[NDArray[Any], List[Any]]] = {}
    df["int_col"] = np_rs.randint(0, 200, size)
    df["float_col"] = np_rs.randn(size)

    df["string_col1"] = [gen_random_string(10) for _ in range(size)]
    df["string_col2"] = [gen_random_string(2) for _ in range(size)]

    df["list_int_col"] = [
        np_rs.randint(-10, 10, np_rs.randint(0, 100)) for _ in range(size)
    ]
    df["list_float_col"] = [
        np_rs.rand(np_rs.randint(0, 10)).astype(np.float32) for _ in range(size)
    ]
    df["list_float_fixed_size_col"] = [
        np_rs.rand(7).astype(np.float32) for _ in range(size)
    ]
    return pd.DataFrame(df)


def generated_partitioned_parquet_file(
    path: str, size: int, n_partitions: int = 20, seed: int = 123
) -> None:
    df = generate_random_pandas_df(size, seed)

    if n_partitions > 0:
        df["part_key"] = np.arange(size) % n_partitions

    table = pa.Table.from_pandas(df)

    pq.write_to_dataset(
        table,
        path,
        partition_cols=["part_key"] if n_partitions > 0 else None,
        existing_data_behavior="delete_matching",
        **{"row_group_size": 110},
    )


@pytest.fixture()
def single_file() -> Generator[str, None, None]:
    tmpdir = tempfile.mkdtemp()
    tmp_parquet_ds_path = os.path.join(tmpdir, "test")
    generated_partitioned_parquet_file(
        tmp_parquet_ds_path, size=10**3, n_partitions=0
    )
    yield tmp_parquet_ds_path
    shutil.rmtree(tmpdir)


@pytest.fixture()
def multi_partition_file() -> Generator[str, None, None]:
    tmpdir = tempfile.mkdtemp()
    tmp_parquet_ds_path = os.path.join(tmpdir, "test")
    generated_partitioned_parquet_file(tmp_parquet_ds_path, size=2 * 10**3)
    yield tmp_parquet_ds_path
    shutil.rmtree(tmpdir)


class TestParquetDataloader:
    def test_simple_dataload(self, multi_partition_file: str) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=multi_partition_file,
            batch_size=11,
            nb_parallel_fragments=2,
            seed=333,
        )
        res: List[pd.DataFrame] = list(parquet_iterator(config))

        assert all(isinstance(x, pa.Table) for x in res)

        assert list(res[0].to_pandas().columns) == [
            "int_col",
            "float_col",
            "string_col1",
            "string_col2",
            "list_int_col",
            "list_float_col",
            "list_float_fixed_size_col",
            "part_key",
        ]

        assert Counter(map(len, res)) == Counter({11: 180, 2: 10})  # 180 * 11
        assert sum(map(len, res)) == 2000

        # determinism check
        config_new = ParquetBasicDataloaderConfig(
            parquet_path=multi_partition_file,
            batch_size=11,
            nb_parallel_fragments=2,
            seed=333,
            output_format=ParquetBatchFormat.pandas,
        )
        res_bis = list(parquet_iterator(config_new))

        assert all(isinstance(x, pd.DataFrame) for x in res_bis)

        assert all(
            (x["float_col"].to_pandas() == y["float_col"]).all()
            for x, y in zip(res, res_bis)
        )

        config_another_seed = ParquetBasicDataloaderConfig(
            parquet_path=multi_partition_file,
            batch_size=11,
            nb_parallel_fragments=2,
            seed=111,
            output_format=ParquetBatchFormat.pandas,
        )
        res_ter = list(parquet_iterator(config_another_seed))
        assert any(
            (x["float_col"] != y["float_col"]).any() for x, y in zip(res, res_ter)
        )

    def test_filtered_with_columns_dataload(self, multi_partition_file: str) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=multi_partition_file,
            batch_size=3,
            nb_parallel_fragments=5,
            seed=111,
            columns=["string_col2", "list_int_col", "float_col"],
            filters=[("float_col", ">", 0)],
            output_format=ParquetBatchFormat.pandas,
        )

        res: List[pd.DataFrame] = list(parquet_iterator(config))

        assert list(res[0].columns) == ["string_col2", "list_int_col", "float_col"]

        assert Counter(map(len, res)) == Counter({3: 340, 1: 2})

    def test_filtered_with_columns_dataload_min_batch_size(
        self, multi_partition_file: str
    ) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=multi_partition_file,
            batch_size=3,
            nb_parallel_fragments=5,
            seed=111,
            min_batch_size=3,
            columns=["string_col2", "list_int_col", "float_col"],
            filters=[("float_col", ">", 0)],
            output_format=ParquetBatchFormat.pandas,
        )
        res = list(parquet_iterator(config))
        assert Counter(map(len, res)) == Counter({3: 340})

    def test_ordered_dataload(self, multi_partition_file: str) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=multi_partition_file,
            batch_size=20,
            nb_parallel_fragments=20,
            order_by_length="list_int_col",
            seed=123,
            output_format=ParquetBatchFormat.pandas,
        )
        res: List[pd.DataFrame] = list(parquet_iterator(config))
        length_by_batches = [tt["list_int_col"].apply(len) for tt in res]
        length_by_batches_diff = max(tt.max() - tt.min() for tt in length_by_batches)
        total_length = sum(map(len, length_by_batches))

        assert length_by_batches_diff < 4
        assert total_length == 2000
        assert all(len(tt) == 20 for tt in length_by_batches)

    def test_ordered_max_token_dataload(self, multi_partition_file: str) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=multi_partition_file,
            nb_parallel_fragments=20,
            order_by_length="list_int_col",
            max_tokens=3000,
            seed=123,
            output_format=ParquetBatchFormat.pandas,
        )
        res: List[pd.DataFrame] = list(parquet_iterator(config))
        length_by_batches = [tt["list_int_col"].apply(len) for tt in res]
        length_by_batches_diff = max(tt.max() - tt.min() for tt in length_by_batches)
        max_padded_total_length = max(tt.max() * len(tt) for tt in length_by_batches)
        mean_padded_total_length = np.mean(
            [tt.max() * len(tt) for tt in length_by_batches]
        )
        total_length = sum(map(len, length_by_batches))

        assert length_by_batches_diff <= 12
        assert total_length == 2000
        assert max_padded_total_length <= 3000
        assert mean_padded_total_length >= 2900

    def test_ordered_max_token_single_file_dataload(self, single_file: str) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=single_file,
            nb_parallel_fragments=2,
            batch_size=10,
            seed=333,
        )
        res: List[pa.Table] = list(parquet_iterator(config))

        assert Counter(map(len, res)) == Counter({10: 100})

        assert res[0].column_names == [
            "int_col",
            "float_col",
            "string_col1",
            "string_col2",
            "list_int_col",
            "list_float_col",
            "list_float_fixed_size_col",
        ]

    def test_dataload_without_shuffle(self, multi_partition_file: str) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=multi_partition_file,
            nb_parallel_fragments=4,
            nb_prefetch=2,
            num_parallel_calls=3,
            shuffle=False,
            batch_size=17,
            columns=["float_col"],
        )
        res = pa.concat_tables(list(parquet_iterator(config)))
        res_relaod = pq.read_table(multi_partition_file, columns=["float_col"])

        assert res.equals(res_relaod)

    def test_dataload_max_row_groups(self, single_file: str) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=single_file,
            nb_parallel_fragments=1,
            nb_prefetch=2,
            num_parallel_calls=3,
            batch_size=250,
        )
        res = list(list(parquet_iterator(config)))

        assert Counter(list(map(len, res))) == Counter({110: 9, 10: 1})

        config = ParquetBasicDataloaderConfig(
            parquet_path=single_file,
            nb_parallel_fragments=2,  # increasing this
            nb_prefetch=2,
            num_parallel_calls=3,
            batch_size=250,
        )
        res = list(list(parquet_iterator(config)))

        assert Counter(list(map(len, res))) == Counter({220: 4, 120: 1})
