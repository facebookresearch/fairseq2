# import polars as pl
# import pyarrow as pa
# import pyarrow.parquet as pq

import fairseq2
from fairseq2.data.parquet.utils import pyarrow_cpu

fairseq2.setup_fairseq2()
from stopes.fb_config import get_filesystem_from_path

from fairseq2.data.text.tokenizers import get_text_tokenizer_hub
from fairseq2.gang import FakeGang
from fairseq2.typing import CPU

path = "/checkpoint/meres/artyomko/data/nllb_data_sharded_fs2/shard=000/"
# path = "blobstore://meresnouserdata/datasets/huggingface/raw/nllb_parquet/"
path, filesystem = get_filesystem_from_path(path)

# ds = pq.ParquetDataset(path, filesystem=filesystem)


from fairseq2.data.parquet.recipes.mt_parallel_text import (
    BiTextColumns,
    ParquetParallelTextDataset,
    ParquetParallelTextDatasetConfig,
)

columns = BiTextColumns(
    source_text="source_text",
    target_text="target_text",
    source_lang="source_lang",
    target_lang="target_lang",
    domain="domain",
)

config = ParquetParallelTextDatasetConfig(
    parquet_path=path,
    columns=columns,
    filesystem=filesystem,
    fragment_shuffle_window=30,
    sample_shuffle_window=20_000,
    max_tokens=14_000,
    direction_batch_size=2,
    direction_weights_manifest_path="/checkpoint/meres/padqn/data/nllb_data_sharded_fs2/shard000/train/MANIFEST",
)

dataset = ParquetParallelTextDataset("nllb", config)

tokenizer_hub = get_text_tokenizer_hub()
tokenizer = tokenizer_hub.load("nllb-200")

loader = dataset.create_reader(
    "train", tokenizer=tokenizer, gang=FakeGang(rank=0, size=1, device=CPU)
)


from time import sleep

from tqdm.auto import tqdm

with pyarrow_cpu(30):
    pbar = tqdm(total=None)
    for bb in tqdm(loader):
        batch = bb[0]
        pbar.update(batch.source_seqs.numel() + batch.target_seqs.numel())
        sleep(0.05)
