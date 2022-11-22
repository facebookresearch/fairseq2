import logging
from typing import Iterable, Iterator, List, Mapping

import datasets

import fairseq2.distributed
from fairseq2.generate import Tokenizer
from fairseq2.typing import Device

from . import Batch

log = logging.getLogger(__name__)


def _finalize_batch(
    tokenizer: Tokenizer,
    src_batch: List[str],
    tgt_batch: List[str],
    src_bos: int,
    tgt_bos: int,
    device: Device,
) -> Batch:
    source = tokenizer.encode_batch(src_batch, bos=src_bos)
    return Batch(
        source=source.to(device),
        target=tokenizer.encode_batch(tgt_batch, bos=tgt_bos).to(device),
        num_tokens=tokenizer.num_tokens(source),
    )


class NllbDataLoader(Iterable[Batch]):
    # TODO: this should be made more generic, here nllb and flores are hardcoded.
    def __init__(
        self,
        src: str,
        tgt: str,
        split: str,
        tokenizer: Tokenizer,
        batch_size: int,
        env: fairseq2.distributed.Env,
    ):
        self.split = split
        self.src = src
        self.tgt = tgt
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.global_rank = env.global_rank
        self.world_size = env.world_size
        self.device = env.device
        self.epoch = 0

        data: Mapping[str, datasets.Dataset] = {}
        assert split in ("train", "valid", "test")
        if split == "train":
            # try both language pairs
            try:
                data = datasets.load_dataset(
                    "allenai/nllb", f"{src}-{tgt}", save_infos=True, streaming=True
                )  # type: ignore
            except Exception:
                data = datasets.load_dataset(
                    "allenai/nllb", f"{tgt}-{src}", save_infos=True, streaming=True
                )  # type: ignore
            self.data = data["train"]
            self.extract_src = lambda sample: sample["translation"][src]
            self.extract_tgt = lambda sample: sample["translation"][tgt]
        else:
            flores_split = "dev" if split == "valid" else "devtest"
            data = datasets.load_dataset("facebook/flores", f"{src}-{tgt}")  # type: ignore
            self.data = data[flores_split]
            self.extract_src = lambda sample: sample["sentence_" + src]
            self.extract_tgt = lambda sample: sample["sentence_" + tgt]
        # HF doesn't allow to shard and stream at the same time ?
        # self.data = self.data.shard(num_shards=self.world_size, index=self.global_rank)

    def __iter__(self) -> Iterator[Batch]:
        if hasattr(self.data, "__len__"):
            log.info(
                f"Starting {self.split} epoch {self.epoch} with {len(self.data)} samples from memory."
            )
        else:
            dataset_size = (
                getattr(self.data._info, "size_in_bytes", 0) / 1024 / 1024 / 1024
            )
            log.info(
                f"Starting {self.split} epoch {self.epoch} with {dataset_size:.1f}Gb of data, streaming from disk."
            )
        src_bos = self.tokenizer.special_tokens[self.src]
        tgt_bos = self.tokenizer.special_tokens[self.tgt]
        src_batch, tgt_batch = [], []
        for i, sample in enumerate(self.data):
            if i % self.world_size != self.global_rank:
                continue

            src_batch.append(self.extract_src(sample))
            tgt_batch.append(self.extract_tgt(sample))
            if len(src_batch) == self.batch_size:
                yield _finalize_batch(
                    self.tokenizer, src_batch, tgt_batch, src_bos, tgt_bos, self.device
                )
                src_batch, tgt_batch = [], []
        if src_batch:
            yield _finalize_batch(
                self.tokenizer, src_batch, tgt_batch, src_bos, tgt_bos, self.device
            )

        log.info(f"End of epoch {self.epoch}, with {i * self.batch_size} samples")
        self.epoch += 1
