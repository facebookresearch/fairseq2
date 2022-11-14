import logging
from typing import Iterable, Iterator, List, Mapping

import datasets
import torch

from fairseq2.generate import Tokenizer

from . import Batch

log = logging.getLogger(__name__)


def _finalize_batch(
    tokenizer: Tokenizer,
    src_batch: List[str],
    tgt_batch: List[str],
    device: torch.device,
) -> Batch:
    source = tokenizer.encode_batch(src_batch)
    return Batch(
        source=source.to(device),
        target=tokenizer.encode_batch(tgt_batch).to(device),
        num_tokens=tokenizer.num_tokens(source),
    )


class NllbDataLoader(Iterable[Batch]):
    # TODO: this should be made more generic, here nllb and flores are hardcoded.
    def __init__(
        self,
        train: bool,
        src: str,
        tgt: str,
        tokenizer: Tokenizer,
        batch_size: int,
        global_rank: int,
        world_size: int,
        device: torch.device,
    ):
        self.train = train
        self.src = src
        self.tgt = tgt
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.global_rank = global_rank
        self.world_size = world_size
        self.device = device
        self.epoch = 0

        data: Mapping[str, datasets.Dataset] = {}
        if train:
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
            data = datasets.load_dataset("facebook/flores", f"{src}-{tgt}")  # type: ignore
            self.data = data["dev"]
            self.extract_src = lambda sample: sample["sentence_" + src]
            self.extract_tgt = lambda sample: sample["sentence_" + tgt]
        # HF doesn't allow to shard and stream at the same time ?
        # self.data = self.data.shard(num_shards=self.world_size, index=self.global_rank)

    def __iter__(self) -> Iterator[Batch]:
        name = "train" if self.train else "valid"
        if hasattr(self.data, "__len__"):
            log.info(
                f"Starting {name} epoch {self.epoch} with {len(self.data)} samples from memory."
            )
        else:
            dataset_size = (
                getattr(self.data._info, "size_in_bytes", 0) / 1024 / 1024 / 1024
            )
            log.info(
                f"Starting {name} epoch {self.epoch} with {dataset_size:.1f}Gb of data, streaming from disk."
            )

        src_batch, tgt_batch = [], []
        for i, sample in enumerate(self.data):
            if i % self.world_size != self.global_rank:
                continue

            src_batch.append(self.extract_src(sample))
            tgt_batch.append(self.extract_tgt(sample))
            if len(src_batch) == self.batch_size:
                yield _finalize_batch(self.tokenizer, src_batch, tgt_batch, self.device)
                src_batch, tgt_batch = [], []
        if src_batch:
            yield _finalize_batch(self.tokenizer, src_batch, tgt_batch, self.device)

        log.info(f"End of epoch {self.epoch}, with {i * self.batch_size} samples")
        self.epoch += 1
