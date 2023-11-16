# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from dataclasses import dataclass

import pyarrow as pa
import torch
from torch import Tensor

from fairseq2.data.audio import WaveformToFbankConverter
from fairseq2.data.data_pipeline import DataPipelineBuilder
from fairseq2.data.text import SentencePieceEncoder
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from fairseq2.typing import Device
from fairseq2.utils.parquet_dataloader import (
    ParquetBasicDataloaderConfig,
    NestedDictTensor,
    ParquetBasicDataLoader,
    ParquetBatchFormat,
    pyarrow_cpu,
)


@dataclass
class SeqsBatch:
    # XXX: using TensorDict would be more practical here
    # for batch level operations like to(device), slicing, del and so on
    src_tokens: tp.Optional[Tensor]
    src_lengths: tp.Optional[Tensor]
    target_tokens: Tensor
    prev_output_tokens: Tensor
    target_lengths: Tensor
    prefix_tokens: tp.Optional[Tensor]

    def __del__(self) -> None:
        """Explicitly delete tensors
        to force GPU memory cleanup"""
        for tensor in [
            self.src_tokens,
            self.src_lengths,
            self.target_tokens,
            self.prev_output_tokens,
            self.target_lengths,
            self.prefix_tokens,
        ]:
            if tensor is not None:
                tensor = tensor.cpu()
                del tensor


def batch_collater(inp: tp.List[Tensor], padding: int) -> tp.Dict[str, Tensor]:
    # TODO: replace it with fairseq2 Collater
    seq_lens = torch.IntTensor([x.shape[0] for x in inp])
    return {
        "seqs": torch.nested.to_padded_tensor(
            torch.nested.as_nested_tensor(inp), padding=padding
        ),
        "seq_lens": seq_lens,
        "is_ragged": False
        if len(seq_lens) == 0
        else (seq_lens != seq_lens[0]).any().item(),
    }


_not_init = object()  # deal with class inheritance


@dataclass
class ASRDataLoadingConfig(ParquetBasicDataloaderConfig):
    text_tokenizer: NllbTokenizer = _not_init  # type: ignore

    audio_column: str = "audio_wav"
    text_column: str = "src_text"
    lang_column: str = "src_lang"

    audio_sample_rate: int = 16_000
    """Audio sample rate of the decoded wavs"""

    fbank_feats_pad_idx: int = 0
    """The pad index to use in fbanks batching."""

    max_audio_frames_per_sample: int = 1500
    """Accept only samples with less than max_audio_frames_per_sample frames in fbanks"""

    device: Device = Device("cpu")
    """The device on which to allocate batches."""

    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        super().__post_init__()

        self.output_format = ParquetBatchFormat.torch

        assert isinstance(self.text_tokenizer, NllbTokenizer)
        assert self.drop_null
        assert self.max_audio_frames_per_sample >= 1

        minimal_columns_set = [self.audio_column, self.text_column, self.lang_column]
        if self.columns is not None:  # extend
            self.columns += [
                col for col in minimal_columns_set if col not in self.columns
            ]
        else:
            self.columns = minimal_columns_set

        max_frame_filter = pa.compute.less_equal(
            pa.compute.list_value_length(pa.dataset.field(self.audio_column)),
            self.audio_sample_rate * self.max_audio_frames_per_sample / 99.6,
        )
        if self.filters is None:
            self.filters = max_frame_filter
        else:  # extend the exsiting filters
            self.filters = pa.compute.and_(self.filters, max_frame_filter)


class ASRBatchIterator:
    """Convenience wrapper for data loading"""

    config: ASRDataLoadingConfig
    raw_parquet_dataloader: ParquetBasicDataLoader

    def __init__(self, config: ASRDataLoadingConfig):
        self.config = config
        self.raw_parquet_dataloader = ParquetBasicDataLoader(self.config)

        self.audio_column, self.text_column, self.lang_column = (
            self.config.audio_column,
            self.config.text_column,
            self.config.lang_column,
        )
        self.vec_audio_column, self.vec_text_column, self.vec_lang_column = (
            f"fbank_{aself.udio_column}",
            f"tokenized_{self.text_column}",
            f"prefixed_{self.lang_column}",
        )

    def __iter__(self) -> tp.Iterable[SeqsBatch]:
        with pyarrow_cpu(self.config.num_parallel_calls):
            yield from iter(
                self.build_asr_datapipeline()
                .prefetch(self.config.num_parallel_calls)
                .and_return(max_num_warnings=4)
            )

    def build_asr_datapipeline(self) -> DataPipelineBuilder:
        converter_to_fbank = self.get_to_fbank_converter()
        tokenizer = SentencePieceEncoder(
            model=self.config.text_tokenizer.model, suffix_tokens=["</s>"]
        )
        prefix_fn = self.get_prefix_lambda()

        def vectorize_batch(batch):
            batch[self.vec_audio_column] = list(
                map(converter_to_fbank, batch.pop(self.audio_column))
            )
            batch[self.vec_text_column] = list(
                map(tokenizer, batch.pop(self.text_column))
            )
            batch[self.vec_lang_column] = list(
                map(prefix_fn, batch.pop(self.lang_column))
            )
            return batch

        def manual_collater(batch):
            batch[self.vec_audio_column] = batch_collater(
                batch[self.vec_audio_column], padding=self.config.fbank_feats_pad_idx
            )
            batch[self.vec_lang_column] = torch.stack(batch[self.vec_lang_column])
            batch[self.vec_text_column] = batch_collater(
                batch[self.vec_text_column],
                padding=self.config.text_tokenizer.vocab_info.pad_idx,
            )
            return batch

        builder = (
            self.raw_parquet_dataloader.build_epoch_iterator_pipeline()
            .map(vectorize_batch, num_parallel_calls=self.config.num_parallel_calls)
            .map(manual_collater, num_parallel_calls=self.config.num_parallel_calls)
            .map(
                self.filter_audio_fbank_nulls,
                num_parallel_calls=self.config.num_parallel_calls,
            )
            .map(
                self.map_to_seqs_batch,
                num_parallel_calls=self.config.num_parallel_calls,
            )
            .filter(
                lambda batch: bool(
                    batch.src_lengths.shape[0] >= self.config.min_batch_size
                )
            )
        )

        return builder

    def filter_audio_fbank_nulls(self, batch: NestedDictTensor) -> NestedDictTensor:
        null_count_per_row = torch.sum(
            torch.isnan(batch[self.vec_audio_column]), dim=[-2, -1]
        )
        if torch.any(null_count_per_row > 0):
            rows_to_keep = null_count_per_row == 0
            batch = {k: v[rows_to_keep] for k, v in batch.items()}

        return batch

    def map_to_seqs_batch(self, batch: NestedDictTensor) -> SeqsBatch:
        """Convert batch tree into a convenient representation"""

        # text tokens
        target_letter_output_tokens = torch.cat(
            (
                batch[self.vec_lang_column][:, 0, :],
                batch[self.vec_text_column]["seqs"],
            ),
            dim=1,
        )
        target_letter_target = target_letter_output_tokens[:, 1:]
        target_letter_prev_output_tokens = torch.clone(target_letter_output_tokens)[
            :, :-1
        ]
        target_letter_prev_output_tokens[
            target_letter_prev_output_tokens
            == self.config.text_tokenizer.vocab_info.eos_idx
        ] = self.config.text_tokenizer.vocab_info.pad_idx  # type:ignore
        target_letter_prev_output_tokens[
            :, 0
        ] = self.config.text_tokenizer.vocab_info.eos_idx  # type:ignore

        target_letter_target_lengths = (
            batch[self.vec_text_column]["seq_lens"] + 1
        )  # count EOS
        target_letter_prefix_tokens = batch[self.vec_lang_column][:, 0, 1:]

        device = self.config.device
        seqs_batch = SeqsBatch(
            src_tokens=batch[self.vec_audio_column]["seqs"].to(device),
            src_lengths=batch[self.vec_audio_column]["seq_lens"].to(device),
            target_tokens=target_letter_target.to(device),
            prev_output_tokens=target_letter_prev_output_tokens.to(device),
            target_lengths=target_letter_target_lengths.to(device),
            prefix_tokens=target_letter_prefix_tokens.to(device),
        )
        return seqs_batch

    def get_to_fbank_converter(self):
        _convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=True,
            device=torch.device("cpu"),
            dtype=self.config.dtype,
        )

        def convert_to_fbank(wav: Tensor) -> Tensor:
            return _convert_to_fbank(
                {
                    "waveform": torch.unsqueeze(wav, 1),
                    "sample_rate": self.config.audio_sample_rate,
                }
            )["fbank"]

        return convert_to_fbank

    def get_prefix_lambda(self):
        def prefix_fn(lang_tok):
            return torch.LongTensor(
                [
                    self.config.text_tokenizer.create_encoder(
                        lang=lang_tok.bytes().decode("utf8"), mode="target"
                    ).prefix_indices.tolist(),  # type:ignore
                ]
            )

        return prefix_fn
