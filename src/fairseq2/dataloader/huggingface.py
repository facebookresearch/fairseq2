import datetime
import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional

import datasets
import torch

import fairseq2.distributed
from fairseq2.generate import SpeechToTextTokenizer, Tokenizer
from fairseq2.typing import DataType, Device

from . import Seq2SeqBatch, Text2TextBatch

log = logging.getLogger(__name__)


def _finalize_batch(
    tokenizer: Tokenizer,
    src_batch: List[str],
    tgt_batch: List[str],
    src_bos: int,
    tgt_bos: int,
    device: Device,
) -> Seq2SeqBatch:
    source = tokenizer.encode_batch(src_batch, bos=src_bos)
    target = tokenizer.encode_batch(tgt_batch, bos=tgt_bos)
    return Seq2SeqBatch(
        source=source.to(device),
        target=target.to(device),
        num_tokens=tokenizer.num_tokens(target),
    )


class NllbDataLoader(Iterable[Seq2SeqBatch]):
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

    def __iter__(self) -> Iterator[Seq2SeqBatch]:
        src_bos = self.tokenizer.special_tokens[self.src]
        tgt_bos = self.tokenizer.special_tokens[self.tgt]

        for batch in self._iter_str():
            yield _finalize_batch(
                self.tokenizer, batch.src, batch.tgt, src_bos, tgt_bos, self.device
            )

    def _iter_str(self) -> Iterator[Text2TextBatch]:
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
        batch = Text2TextBatch([], [])
        for i, sample in enumerate(self.data):
            if i % self.world_size != self.global_rank:
                continue

            batch.src.append(self.extract_src(sample))
            batch.tgt.append(self.extract_tgt(sample))
            if len(batch.src) == self.batch_size:
                yield batch
                batch = Text2TextBatch([], [])
        if batch.src:
            yield batch

        log.info(f"End of epoch {self.epoch}, with {i * self.batch_size} samples")
        self.epoch += 1

    @staticmethod
    def combine_and_dump(
        src: str, tgt: str, split: str, output: Path, limit: int = 0
    ) -> None:
        env = fairseq2.distributed.Env(output.parent, 0, 1, Device("cpu"))
        loader = NllbDataLoader(
            src,
            tgt,
            split,
            tokenizer=None,  # type: ignore
            batch_size=16,
            env=env,
        )
        i = 0
        with output.open("wt") as o:
            for batch in loader._iter_str():
                for src in batch.src:
                    print(src, file=o)
                for tgt in batch.tgt:
                    print(tgt, file=o)
                i += 2 * len(batch.src)
                if 0 < limit < i:
                    break


class AsrBatch:
    def __init__(self, sampling_rate: int):
        # Actually those are numpy array
        self.audio_features: List[torch.Tensor] = []
        self.sentences: List[str] = []
        self.batch_len: int = 0
        self.sampling_rate: int = sampling_rate

    def append(self, audio: torch.Tensor, sentence: str) -> int:
        self.audio_features.append(audio)
        self.sentences.append(sentence)
        self.batch_len += len(audio)
        return self.batch_len


class AsrDataloader(Iterable[Seq2SeqBatch]):
    def __init__(
        self,
        name: str,
        language: str,
        split: str,
        tokenizer: SpeechToTextTokenizer,
        *,
        batch_size: int = 0,
        batch_duration: Optional[datetime.timedelta] = None,
        env: fairseq2.distributed.Env,
        dtype: DataType,
    ):
        self.tokenizer = tokenizer
        self.sampling_rate = self.tokenizer.sampling_rate
        self.env = env
        self.dtype = dtype
        if batch_size > 0:
            self.batch_size = batch_size
        else:
            assert (
                batch_duration is not None
            ), "Need to specify either batch_size or batch_duration"
            self.batch_size = int(batch_duration.total_seconds() * self.sampling_rate)

        self.data = (
            datasets.load_dataset(name, language, split=split).remove_columns(
                [
                    "accent",
                    "age",
                    "client_id",
                    "down_votes",
                    "gender",
                    "locale",
                    "path",
                    "segment",
                    "up_votes",
                ]
            )
            # Downsampling
            .cast_column("audio", datasets.Audio(sampling_rate=(self.sampling_rate)))
        )
        self.epoch = 0

    def _finalize_batch(self, batch: AsrBatch) -> Seq2SeqBatch:
        device = self.env.device
        target = self.tokenizer.encode_batch(batch.sentences)
        return Seq2SeqBatch(
            self.tokenizer.encode_audio(
                batch.audio_features, sampling_rate=batch.sampling_rate
            ).to(device=device, dtype=self.dtype),
            target.to(device),
            self.tokenizer.num_tokens(target),
        )

    def __iter__(self) -> Iterator[Seq2SeqBatch]:
        batch = AsrBatch(self.sampling_rate)
        _, global_rank, world_size, device = self.env

        for i, sample in enumerate(self.data):
            if i % world_size != global_rank:
                continue

            audio = sample["audio"]
            # We already resampled earlier
            assert audio["sampling_rate"] == self.sampling_rate
            batch_len = batch.append(audio["array"], sample["sentence"])

            if batch_len >= self.batch_size:
                yield self._finalize_batch(batch)
                batch = AsrBatch(self.sampling_rate)
        if batch.audio_features:
            yield self._finalize_batch(batch)

        log.info(f"End of epoch {self.epoch}, with {i * self.batch_size} samples")
        self.epoch += 1
