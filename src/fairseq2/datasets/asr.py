# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Set, cast, final

import torch
from torch import Tensor
from torch.nn.functional import layer_norm

from fairseq2.assets import AssetCard
from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    FileMapper,
    SequenceData,
    create_bucket_sizes,
    read_sequence,
)
from fairseq2.data.audio import AudioDecoder
from fairseq2.data.text import (
    StrSplitter,
    TextTokenizer,
    default_raw_sentencepiece_tokenizer_loader,
    load_text_tokenizer,
    read_text,
)
from fairseq2.datasets.data_reader import DataPipelineReader, DataReader
from fairseq2.datasets.error import DatasetError
from fairseq2.datasets.loader import AbstractDatasetLoader, DelegatingDatasetLoader
from fairseq2.gang import Gang
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import DataType, override


class AsrDataset(ABC):
    """Represents an automatic speech recognition dataset."""

    @abstractmethod
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_audio_len: int,
        max_num_elements: int,
        *,
        dtype: DataType = torch.float32,
        min_audio_len: int = 1,
        normalize_audio: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        **extras: Any,
    ) -> DataReader[Seq2SeqBatch]:
        """Create a dataset reader.

        :param split:
            The split to read.
        :param tokenizer:
            The tokenizer to encode target text.
        :param gang:
            The gang over which to shard the dataset.
        :param max_audio_len:
            The maximum audio length of each example. Examples longer than
            this value will be cropped.
        :param max_num_elements:
            The maximum number of elements in each batch.
        :param dtype:
            The data type of the decoded audio sequences.
        :param min_audio_len:
            The minimum audio length of each example. Examples shorter than
            this value will be dropped.
        :param normalize_audio:
            If ``True``, normalizes audio to have zero mean and unit variance.
        :param example_shuffle_window:
            The size of the sliding window for shuffling examples. If ``1``, no
            shuffling is performed; if ``0``, true shuffling is performed by
            loading the entire dataset.
        :param batch_shuffle_window:
            The size of the sliding window for shuffling batches. If ``1``, no
            shuffling is performed; if ``0``, true shuffling is performed by
            loading the entire dataset.
        :param num_accumulate:
            The number of batches to accumulate in each iteration. Typically
            used with gradient accumulation during training.
        :param num_prefetch:
            The number of batches to prefetch in background.
        :param seed:
            The seed to initialize the random number generators used internally.
        :param extras:
            The extra parameters specific to the dataset implementation.
        """

    @abstractmethod
    def splits(self) -> Set[str]:
        """Return the set of splits."""


# TODO: FIX, INFER
npc = 10


# TODO: Work in progress!
@final
class GenericAsrDataset(AsrDataset):
    """Represents a generic manifest-based ASR dataset."""

    _dataset_name: str
    _manifest_dir: Path
    _splits: Set[str]

    def __init__(self, dataset_name: str, manifest_dir: Path) -> None:
        """
        :param dataset_name:
            The name of the dataset.
        :param manifest_dir:
            The directory under which the manifest files resides.
        """
        self._dataset_name = dataset_name
        self._manifest_dir = manifest_dir

        self._splits = set()

        for tsv_file in manifest_dir.glob("*.tsv"):
            self._splits.add(tsv_file.stem)

    @override
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_audio_len: int,
        max_num_elements: int,
        *,
        dtype: DataType = torch.float32,
        min_audio_len: int = 1,
        normalize_audio: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        cached_fd_count: int = 1000,
        **extras: Any,
    ) -> DataPipelineReader[Seq2SeqBatch]:
        """
        :param cached_fd_count:
            The maximum number of file descriptors to keep open while reading
            audio files.
        """
        if split not in self._splits:
            raise ValueError(
                f"`split` must be a valid split name, but the {self._dataset_name} dataset has no split named '{split}'."
            )

        root_data_dir = self._retrieve_data_directory(split)

        manifest = self._load_manifest(split)

        builder = read_sequence(manifest)

        # Shuffle examples. Must be consistent across all processes.
        if example_shuffle_window != 1:
            builder.shuffle(example_shuffle_window, seed=seed)

        seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        # Bucket by audio length.
        bucket_sizes = create_bucket_sizes(
            max_num_elements=max_num_elements,
            max_seq_len=max_audio_len,
            min_seq_len=min_audio_len,
            num_seqs_multiple_of=8,
        )

        builder.bucket_by_length(
            bucket_sizes,
            selector="audio_size",
            min_data_len=min_audio_len,
            skip_below_min_examples=True,
            skip_above_max_examples=True,
        )

        # Shuffle buckets.
        if batch_shuffle_window != 1:
            builder.shuffle(batch_shuffle_window, seed=seed)

        seed += 1

        # Memory map audio files.
        file_mapper = FileMapper(root_data_dir, cached_fd_count=cached_fd_count)

        builder.map(file_mapper, selector="[*].audio")

        # Decode audio.
        audio_decoder = AudioDecoder(dtype=torch.float32 if normalize_audio else dtype)

        builder.map(audio_decoder, selector="[*].audio.data")

        # TODO(balioglu): Check/adjust sample size

        # Normalize audio if requested.
        def normalize(waveform: Tensor) -> Tensor:
            with torch.no_grad():
                waveform = layer_norm(waveform, waveform.shape)

            return waveform.to(dtype)

        if normalize_audio:
            builder.map(normalize, selector="[*].audio.data.waveform")

        # Tokenize target text.
        text_encoder = tokenizer.create_encoder()

        builder.map(text_encoder, selector="[*].text", num_parallel_calls=npc)

        # Collate bucketed examples into a batch.
        text_collate_opts = CollateOptionsOverride(
            "text", pad_value=tokenizer.vocab_info.pad_idx
        )

        collater = Collater(pad_value=0, overrides=[text_collate_opts])

        builder.map(collater, num_parallel_calls=npc)

        # Prefetch `num_prefetch` examples in background.
        builder.prefetch(num_prefetch)

        # Wrap examples with `Seq2SeqBatch`.
        def _example_to_batch(example: Dict[str, Any]) -> Seq2SeqBatch:
            source_data = cast(SequenceData, example["audio"]["data"]["waveform"])
            target_data = cast(SequenceData, example["text"])

            source_seqs, source_padding_mask = get_seqs_and_padding_mask(
                source_data, gang.device
            )
            target_seqs, target_padding_mask = get_seqs_and_padding_mask(
                target_data, gang.device
            )

            return Seq2SeqBatch(
                source_seqs,
                source_padding_mask,
                target_seqs,
                target_padding_mask,
                example,
            )

        pipeline = builder.map(_example_to_batch).and_return()

        return DataPipelineReader[Seq2SeqBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=False,
            sync_batches=True,
        )

    def _retrieve_data_directory(self, split: str) -> Path:
        tsv_file = self._manifest_dir.joinpath(f"{split}.tsv")

        try:
            with tsv_file.open() as fp:
                line = fp.readline().rstrip()
        except OSError as ex:
            raise DatasetError(
                f"The manifest file '{tsv_file}' of the {self._dataset_name} dataset cannot be read. See nested exception for details."
            ) from ex

        try:
            return Path(line)
        except ValueError:
            raise DatasetError(
                f"The first line of the manifest file '{tsv_file}' of the {self._dataset_name} dataset must point to a data directory."
            )

    def _load_manifest(self, split: str) -> List[Any]:
        def build_tsv_pipeline() -> DataPipeline:
            tsv_file = self._manifest_dir.joinpath(f"{split}.tsv")

            builder = read_text(tsv_file, rtrim=True, memory_map=True)

            builder.skip(1)

            field_splitter = StrSplitter(names=["audio", "audio_size"])

            builder.map(field_splitter, num_parallel_calls=npc)

            return builder.and_return()

        def build_wrd_pipeline() -> DataPipeline:
            wrd_file = self._manifest_dir.joinpath(f"{split}.wrd")

            builder = read_text(wrd_file, key="text", rtrim=True, memory_map=True)

            return builder.and_return()

        tsv_pipeline = build_tsv_pipeline()
        wrd_pipeline = build_wrd_pipeline()

        builder = DataPipeline.zip([tsv_pipeline, wrd_pipeline], flatten=True)

        # Cast audio size to integer.
        builder.map(int, selector="audio_size")

        # TODO: Use `cache()` op.
        return list(builder.and_return())

    @override
    def splits(self) -> Set[str]:
        return self._splits


load_asr_dataset = DelegatingDatasetLoader[AsrDataset]()


@final
class GenericAsrDatasetLoader(AbstractDatasetLoader[GenericAsrDataset]):
    @override
    def _load(self, path: Path, card: AssetCard) -> GenericAsrDataset:
        return GenericAsrDataset(card.name, path)


load_generic_asr_dataset = GenericAsrDatasetLoader()

load_librispeech_asr_tokenizer = default_raw_sentencepiece_tokenizer_loader


def _register_asr() -> None:
    load_asr_dataset.register("generic_asr", load_generic_asr_dataset)

    load_text_tokenizer.register("librispeech_asr", load_librispeech_asr_tokenizer)
