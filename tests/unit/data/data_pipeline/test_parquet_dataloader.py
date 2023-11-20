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
import typing as tp
import unittest
from collections import Counter

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fairseq2.utils.parquet_dataloader import (
    ParquetBasicDataLoader,
    ParquetBasicDataloaderConfig,
    ParquetBatchFormat,
)


def gen_random_string(length: int) -> str:
    return "".join(
        random.choice(string.ascii_letters + string.digits) for n in range(length)
    )


def generate_random_pandas_df(size: int, seed: int = 123) -> pd.DataFrame:
    np_rs = np.random.RandomState(seed)
    df: tp.Dict[str, tp.Union[npt.NDArray[tp.Any], tp.List[tp.Any]]] = {}
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


class TestParquetDataloader(unittest.TestCase):
    _tmpdir: str
    _tmp_parquet_ds_path: str
    _tmp_parquet_single_path: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.mkdtemp()
        cls._tmp_parquet_ds_path = os.path.join(cls._tmpdir, "test")
        generated_partitioned_parquet_file(cls._tmp_parquet_ds_path, size=2 * 10**3)

        cls._tmp_parquet_single_path = os.path.join(cls._tmpdir, "single_test.parquet")
        generated_partitioned_parquet_file(
            cls._tmp_parquet_single_path, size=10**3, n_partitions=0
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # Cleanup temp working dir.
        if cls._tmpdir is not None:
            shutil.rmtree(cls._tmpdir)  # type: ignore

    def test_simple_dataload(self) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=self._tmp_parquet_ds_path,
            batch_size=11,
            nb_producers=2,
            seed=333,
        )
        pbdl = ParquetBasicDataLoader(config)
        res: tp.List[pd.DataFrame] = list(iter(pbdl))

        for x in res:
            self.assertIsInstance(x, pa.Table)

        self.assertEqual(
            list(res[0].to_pandas().columns),
            [
                "int_col",
                "float_col",
                "string_col1",
                "string_col2",
                "list_int_col",
                "list_float_col",
                "list_float_fixed_size_col",
                "part_key",
            ],
        )
        self.assertEqual(Counter(map(len, res)), Counter({11: 180, 2: 10}))  # 180 * 11
        self.assertEqual(sum(map(len, res)), 2000)

        # determinism check
        config_new = ParquetBasicDataloaderConfig(
            parquet_path=self._tmp_parquet_ds_path,
            batch_size=11,
            nb_producers=2,
            seed=333,
            output_format=ParquetBatchFormat.pandas,
        )
        pbdl_new = ParquetBasicDataLoader(config_new)
        res_bis = list(iter(pbdl_new))

        for x in res_bis:
            self.assertIsInstance(x, pd.DataFrame)

        self.assertTrue(
            all(
                (x["float_col"].to_pandas() == y["float_col"]).all()
                for x, y in zip(res, res_bis)
            )
        )

        config_another_seed = ParquetBasicDataloaderConfig(
            parquet_path=self._tmp_parquet_ds_path,
            batch_size=11,
            nb_producers=2,
            seed=111,
            output_format=ParquetBatchFormat.pandas,
        )
        pbdl_ter = ParquetBasicDataLoader(config_another_seed)
        res_ter = list(iter(pbdl_ter))
        self.assertTrue(
            any((x["float_col"] != y["float_col"]).any() for x, y in zip(res, res_ter))
        )

    def test_filtered_with_columns_dataload(self) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=self._tmp_parquet_ds_path,
            batch_size=3,
            nb_producers=5,
            seed=111,
            columns=["string_col2", "list_int_col", "float_col"],
            filters=[("float_col", ">", 0)],
            output_format=ParquetBatchFormat.pandas,
        )
        pbdl = ParquetBasicDataLoader(config)
        res: tp.List[pd.DataFrame] = list(iter(pbdl))

        self.assertEqual(
            list(res[0].columns), ["string_col2", "list_int_col", "float_col"]
        )
        self.assertEqual(Counter(map(len, res)), Counter({3: 340, 1: 2}))

    def test_filtered_with_columns_dataload_min_batch_size(self) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=self._tmp_parquet_ds_path,
            batch_size=3,
            nb_producers=5,
            seed=111,
            min_batch_size=3,
            columns=["string_col2", "list_int_col", "float_col"],
            filters=[("float_col", ">", 0)],
            output_format=ParquetBatchFormat.pandas,
        )
        pbdl = ParquetBasicDataLoader(config)
        res = list(iter(pbdl))
        self.assertEqual(Counter(map(len, res)), Counter({3: 340}))

    def test_ordered_dataload(self) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=self._tmp_parquet_ds_path,
            batch_size=20,
            nb_producers=20,
            order_by="list_int_col",
            seed=123,
            output_format=ParquetBatchFormat.pandas,
        )
        pbdl = ParquetBasicDataLoader(config)
        res: tp.List[pd.DataFrame] = list(iter(pbdl))
        length_by_batches = [tt["list_int_col"].apply(len) for tt in res]
        length_by_batches_diff = max(tt.max() - tt.min() for tt in length_by_batches)
        total_length = sum(map(len, length_by_batches))

        self.assertLess(length_by_batches_diff, 4)
        self.assertEqual(total_length, 2000)
        self.assertTrue(all(len(tt) == 20 for tt in length_by_batches))

    def test_ordered_max_token_dataload(self) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=self._tmp_parquet_ds_path,
            nb_producers=20,
            order_by="list_int_col",
            max_tokens=3000,
            seed=123,
            output_format=ParquetBatchFormat.pandas,
        )
        pbdl = ParquetBasicDataLoader(config)
        res: tp.List[pd.DataFrame] = list(iter(pbdl))
        length_by_batches = [tt["list_int_col"].apply(len) for tt in res]
        length_by_batches_diff = max(tt.max() - tt.min() for tt in length_by_batches)
        max_padded_total_length = max(tt.max() * len(tt) for tt in length_by_batches)
        mean_padded_total_length = np.mean(
            [tt.max() * len(tt) for tt in length_by_batches]
        )
        total_length = sum(map(len, length_by_batches))

        self.assertLessEqual(length_by_batches_diff, 12)
        self.assertEqual(total_length, 2000)
        self.assertLessEqual(max_padded_total_length, 3000)
        self.assertGreater(mean_padded_total_length, 2900)

    def test_ordered_max_token_single_file_dataload(self) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=self._tmp_parquet_single_path,
            nb_producers=2,
            batch_size=10,
            seed=333,
        )
        pbdl = ParquetBasicDataLoader(config)
        res: tp.List[pa.Table] = list(iter(pbdl))

        self.assertEqual(Counter(map(len, res)), Counter({10: 100}))
        self.assertEqual(
            res[0].column_names,
            [
                "int_col",
                "float_col",
                "string_col1",
                "string_col2",
                "list_int_col",
                "list_float_col",
                "list_float_fixed_size_col",
            ],
        )

    def test_dataload_without_shuffle(self) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=self._tmp_parquet_ds_path,
            nb_producers=4,
            nb_prefetch=2,
            num_parallel_calls=3,
            shuffle=False,
            batch_size=17,
            columns=["float_col"],
        )
        pbdl = ParquetBasicDataLoader(config)
        res = pa.concat_tables(list(iter(pbdl)))
        res_relaod = pq.read_table(self._tmp_parquet_ds_path, columns=["float_col"])

        self.assertTrue(res.equals(res_relaod))

    def test_dataload_max_row_groups(self) -> None:
        config = ParquetBasicDataloaderConfig(
            parquet_path=self._tmp_parquet_single_path,
            nb_producers=1,
            nb_prefetch=2,
            num_parallel_calls=3,
            batch_size=250,
        )
        pbdl = ParquetBasicDataLoader(config)
        res = list(iter(pbdl))

        self.assertEqual(Counter(list(map(len, res))), Counter({110: 9, 10: 1}))

        config = ParquetBasicDataloaderConfig(
            parquet_path=self._tmp_parquet_single_path,
            nb_producers=2,  # increasing this
            nb_prefetch=2,
            num_parallel_calls=3,
            batch_size=250,
        )
        pbdl = ParquetBasicDataLoader(config)
        res = list(iter(pbdl))

        self.assertEqual(Counter(list(map(len, res))), Counter({220: 4, 120: 1}))
