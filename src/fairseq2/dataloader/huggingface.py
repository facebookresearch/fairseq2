import datetime
import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Tuple

import datasets  # type: ignore[import]
import torch
from torch import Tensor
from transformers import SequenceFeatureExtractor  # type: ignore[import]

import fairseq2.distributed
from fairseq2.data.text import Tokenizer

from . import Seq2SeqBatch, Text2TextBatch

log = logging.getLogger(__name__)


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
        self.batch_size = batch_size
        self.global_rank = env.global_rank
        self.world_size = env.world_size
        self.device = env.device
        self.epoch = 0

        if tokenizer:
            self.pad_idx = tokenizer.vocab_info.pad_idx

            task = "translation"

            self.src_encoder = tokenizer.create_encoder(
                task, lang=src, mode="source", device=env.device, pin_memory=True
            )
            self.tgt_encoder = tokenizer.create_encoder(
                task, lang=tgt, mode="target", device=env.device, pin_memory=True
            )

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

    def _num_tokens(self, tokens: Tensor) -> Tensor:
        return (tokens != self.pad_idx).sum(dim=-1)

    def __iter__(self) -> Iterator[Seq2SeqBatch]:
        for batch in self._iter_str():
            source = self.src_encoder(batch.src)
            target = self.tgt_encoder(batch.tgt)

            yield Seq2SeqBatch(
                source, self._num_tokens(source), target, self._num_tokens(target)
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
        batch_src, batch_tgt = [], []
        for i, sample in enumerate(self.data):
            if i % self.world_size != self.global_rank:
                continue

            batch_src.append(self.extract_src(sample))
            batch_tgt.append(self.extract_tgt(sample))
            if len(batch_src) == self.batch_size:
                yield Text2TextBatch(batch_src, batch_tgt)
                batch_src, batch_tgt = [], []
        if batch_src:
            yield Text2TextBatch(batch_src, batch_tgt)

        log.info(f"End of epoch {self.epoch}, with {i * self.batch_size} samples")
        self.epoch += 1

    @staticmethod
    def combine_and_dump(
        src: str, tgt: str, split: str, output: Path, limit: int = 0
    ) -> None:
        env = fairseq2.distributed.Env(0, 1, torch.device("cpu"))
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
        self.audio_features: List[Tensor] = []
        self.sentences: List[str] = []
        self.batch_len: int = 0
        self.sampling_rate: int = sampling_rate

    def append(self, audio: Tensor, sentence: str) -> int:
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
        feature_extractor: "SequenceFeatureExtractor",
        tokenizer: Tokenizer,
        *,
        batch_size: int = 0,
        batch_duration: Optional[datetime.timedelta] = None,
        env: fairseq2.distributed.Env,
        dtype: torch.dtype,
    ):
        self.feature_extractor = feature_extractor
        self.sampling_rate = feature_extractor.sampling_rate
        self.env = env
        self.device = env.device
        self.dtype = dtype

        self.pad_idx = tokenizer.vocab_info.pad_idx

        self.token_encoder = tokenizer.create_encoder(
            task="transcribe",
            lang=language,
            mode="target",
            device=env.device,
            pin_memory=True,
        )

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
        target = self.token_encoder(batch.sentences)
        source, src_seq_lens = self._encode_audio(batch.audio_features)
        return Seq2SeqBatch(
            source,
            src_seq_lens,
            target.to(device),
            self._num_tokens(target),
        )

    def _encode_audio(self, raw_speech: List[torch.Tensor]) -> Tuple[Tensor, Tensor]:
        features = [
            self.feature_extractor(
                raw_speech=example,
                return_tensors="pt",
                sampling_rate=self.sampling_rate,
            )["input_features"]
            for example in raw_speech
        ]
        src_seq_lens = torch.tensor([f.size(-1) for f in features], device=self.device)
        source = self.feature_extractor.pad(
            {"input_features": [f.squeeze(0) for f in features]},
            padding=True,
            pad_to_multiple_of=128,
            return_attention_mask=False,
            return_tensors="pt",
        )["input_features"]

        assert isinstance(source, Tensor)
        assert source.shape[0] == len(raw_speech)
        source = source.to(device=self.device, dtype=self.dtype)
        return source, src_seq_lens

    def _num_tokens(self, tokens: Tensor) -> Tensor:
        return (tokens != self.pad_idx).sum(dim=-1)

    def __iter__(self) -> Iterator[Seq2SeqBatch]:
        batch = AsrBatch(self.sampling_rate)
        env = self.env

        for i, sample in enumerate(self.data):
            if i % env.world_size != env.global_rank:
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
