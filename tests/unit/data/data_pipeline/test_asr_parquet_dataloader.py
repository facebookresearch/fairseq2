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

from fairseq2.utils.asr_parquet_dataloader import (
    ASRDataLoadingConfig,
    ASRBatchIterator,
    SeqsBatch,
)
from fairseq2.models.nllb import load_nllb_tokenizer


def gen_random_string(length: int) -> str:
    return "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(length)
    )


def generated_partitioned_parquet_file(path: str, size: int, seed: int = 123) -> None:
    np_rs = np.random.RandomState(seed)
    df: tp.Dict[str, tp.Union[npt.NDArray[tp.Any], tp.List[tp.Any]]] = {}

    df["src_text"] = [gen_random_string(np_rs.randint(10, 50)) for _ in range(size)]
    df["src_lang"] = np_rs.choice(["eng_Latn", "deu_Latn", "fra_Latn"], size)
    df["audio_wav"] = [
        np_rs.rand(np_rs.randint(10**4, 3 * 10**4)).astype(np.float32)
        for _ in range(size)
    ]

    table = pa.Table.from_pandas(pd.DataFrame(df))

    pq.write_to_dataset(
        table,
        path,
        partition_cols=["src_lang"],
        existing_data_behavior="delete_matching",
        **{"row_group_size": 110},
    )


class TestASRParquetDataloader(unittest.TestCase):
    _tmpdir: str
    _tmp_parquet_ds_path: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.mkdtemp()
        cls._tmp_parquet_ds_path = os.path.join(cls._tmpdir, "test")
        generated_partitioned_parquet_file(cls._tmp_parquet_ds_path, size=2 * 10**3)

        cls._tokenizer = load_nllb_tokenizer(  # type: ignore
            "nllb-200_dense_distill_600m", progress=False
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # Cleanup temp working dir.
        if cls._tmpdir is not None:
            shutil.rmtree(cls._tmpdir)  # type: ignore

    def test_basic_asr_dataload(self) -> None:
        asr_config = ASRDataLoadingConfig(
            parquet_path=self._tmp_parquet_ds_path,
            batch_size=5,
            order_by="audio_wav",
            text_tokenizer=self._tokenizer,  # type: ignore
            nb_producers=4,
            num_parallel_calls=2,
        )

        asr_iter = ASRBatchIterator(asr_config)

        batches = list(iter(asr_iter))

        lengths = [len(rr.target_lengths) for rr in batches]
        types = [type(rr) for rr in batches]

        self.assertEqual(Counter(lengths), Counter({5: 399, 4: 1, 1: 1}))
        self.assertEqual(Counter(types), Counter({SeqsBatch: 401}))
