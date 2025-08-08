# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
wav2vec2 training dataset - pure data loading logic.

MIGRATION NOTES:
- Core logic from fairseq2:e9fbd6/src/fairseq2/datasets/speech.py
- Maintains 1:1 numerical parity with GenericSpeechDataset.create_reader()
- All preprocessing logic moved to preprocessing.py for clean separation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, Final, Optional, final

import torch

from fairseq2.data.data_pipeline import (
    Collater,
    DataPipelineBuilder,
)
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.datasets import (
    DataPipelineReader,
    DataReader,
    DataReadError,
    DatasetOpenError,
    SequenceBatch,
    SyncMode,
)
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.recipe.config import DatasetSection

from .batch_utils import BatchingPipeline, BatchingStrategy, create_sequence_batch
from .preprocessing import AudioProcessingPipeline

WAV2VEC2_DATASET: Final = "wav2vec2_train"


@final
class Wav2Vec2TrainDataset:
    """
    wav2vec2 training dataset with full v0.4 feature parity.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:294-593
    Class: GenericSpeechDataset(ManifestDatasetInterface, SpeechDataset)
    """

    _name: str
    _manifest_dir: Path  # ORIGINAL: ManifestDatasetInterface._manifest_dir
    _splits: set[str]  # ORIGINAL: ManifestDatasetInterface._splits

    def __init__(self, name: str, manifest_dir: Path, splits: set[str]) -> None:
        """ORIGINAL: ManifestDatasetInterface.__init__() line 193-198"""
        self._name = name
        self._manifest_dir = manifest_dir
        self._splits = splits

    @classmethod
    def from_path(cls, path: Path, name: str) -> "Wav2Vec2TrainDataset":
        """
        Create dataset from path (file or directory).

        ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:200-219
        Method: ManifestDatasetInterface.from_path()
        """
        path = path.expanduser().resolve()  # ORIGINAL: line 201

        if not path.is_dir():  # ORIGINAL: line 203
            return cls(
                name, manifest_dir=path.parent, splits={path.stem}
            )  # ORIGINAL: line 204

        try:
            splits = {f.stem for f in path.glob("*.tsv")}  # ORIGINAL: line 207
        except OSError as ex:
            raise DatasetOpenError(
                name,
                f"The splits under the '{path}' directory of the '{name}' dataset cannot be determined. See the nested exception for details.",  # ORIGINAL: line 210-211
            ) from ex

        if not splits:  # NEW: Added validation
            raise DatasetOpenError(name, f"No .tsv files found in {path}")

        return cls(name, manifest_dir=path, splits=splits)  # ORIGINAL: line 219

    def _retrieve_audio_directory(self, split: str) -> Path | None:
        """
        Retrieve audio directory from manifest file header.
        Expecting the following structure:

        ```text (train-clean-100.tsv)
        /path-to-librispeech/librispeech/062419
        train-clean-100/1553/140047/1553-140047-0000.flac       180080
        train-clean-100/1553/140047/1553-140047-0001.flac       219840
        (...)
        ```

        ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:221-243
        Method: _retrieve_data_directory() (static method)
        """
        manifest_file = self._manifest_dir.joinpath(
            f"{split}.tsv"
        )  # ORIGINAL: line 225

        try:
            with manifest_file.open(encoding="utf-8") as fp:  # ORIGINAL: line 227
                header = fp.readline().rstrip()  # ORIGINAL: line 228
        except OSError as ex:
            raise DataReadError(
                self._name,
                split,
                f"The {manifest_file} manifest file cannot be read. See the nested exception for details.",  # ORIGINAL: line 230-231
            ) from ex

        try:
            audio_dir = Path(header)  # ORIGINAL: line 235
            if audio_dir.exists():  # ORIGINAL: line 236
                return audio_dir  # ORIGINAL: line 238
            return None  # ORIGINAL: line 239
        except ValueError:
            raise DataReadError(
                self._name,
                split,
                f"The first line of {manifest_file} must point to a data directory.",  # ORIGINAL: line 242
            ) from None

    def _read_manifest(
        self, split: str, audio_dir: Path | None, min_audio_len: int, max_audio_len: int
    ) -> DataPipelineBuilder:
        """
        Read and parse TSV manifest file.
        Expecting the following structure:

        ```text (train-clean-100.tsv)
        /path-to-librispeech/librispeech/062419
        train-clean-100/1553/140047/1553-140047-0000.flac       180080
        train-clean-100/1553/140047/1553-140047-0001.flac       219840
        (...)
        ```

        ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:400-420
        In: GenericSpeechDataset._read_manifest()
        """
        tsv_file = self._manifest_dir.joinpath(f"{split}.tsv")  # ORIGINAL: line 400

        builder = read_text(
            tsv_file,
            rtrim=True,
            memory_map=True,
            block_size=10 * 1024 * 1024,  # ORIGINAL: line 401-402
        )

        if audio_dir is not None:  # ORIGINAL: line 404
            builder.skip(1)  # Path to the data directory. ORIGINAL: line 405

        field_splitter = StrSplitter(
            names=["audio", "audio_size"]
        )  # ORIGINAL: line 407
        builder.map(field_splitter)  # ORIGINAL: line 409

        # Convert audio_size to int and clamp to max_audio_len - ORIGINAL: line 411-414
        builder.map(
            lambda x: min(int(x), max_audio_len),
            selector="audio_size",
        )

        # Filter by minimum length - ORIGINAL: line 416
        builder.filter(lambda sample: sample["audio_size"] >= min_audio_len)

        return builder

    def create_reader(
        self,
        split: str,
        gangs: Gangs,
        min_audio_len: int,
        max_audio_len: int,
        *,
        # Batching - ORIGINAL: SpeechReadOptions parameters
        batching_strategy: BatchingStrategy = BatchingStrategy.LENGTH,  # NEW: replaces batching parameter
        batch_size: int | None = None,  # NEW: for STATIC strategy
        max_num_elements: int = 1_500_000,  # ORIGINAL: LengthBatching.max_num_elements
        num_seqs_multiple_of: int = 8,  # ORIGINAL: LengthBatching used this implicitly from GenericSpeechDataset._get_num_seqs_multiple_of
        # Audio processing - ORIGINAL: SpeechReadOptions parameters
        dtype: torch.dtype = torch.float32,
        normalize_audio: bool = False,
        use_fbank: bool = False,
        no_padding: bool = True,
        npc: int = 10,
        # SpecAugment - ORIGINAL: SpeechReadOptions parameters
        spec_aug_p: Optional[float] = None,
        spec_aug_freq_mask_param: int = 80,
        spec_aug_time_mask_param: int = 80,
        # Shuffling and batching - ORIGINAL: SpeechReadOptions parameters
        example_shuffle_window: int = 500_000,
        batch_shuffle_window: int = 0,
        # Performance - ORIGINAL: DataReadOptions parameters
        num_accumulate: int = 1,
        num_prefetch: int = 4,
        drop_remainder: bool = False,
        sync_batches: bool = True,  # ORIGINAL: DataReadOptions line 73
        sync_mode: SyncMode = SyncMode.UNTIL_FIRST,  # ORIGINAL: DataReadOptions line 82
        seed: int = 2,
        max_num_batches: int | None = None,
        cached_fd_count: int = 1000,  # NEW: Hidden in defaults of FileMapper
    ) -> DataReader[SequenceBatch]:
        """
        Create data reader with complete audio processing pipeline.

        ORIGINAL: fairseq2:e9fbd6/src/fairseq2/datasets/speech.py:522-593
        Method: GenericSpeechDataset.create_reader()
        """

        if split not in self._splits:  # ORIGINAL: line 531
            raise ValueError(
                f"Unknown split '{split}'. Available: {self._splits}"
            )  # NEW: better error

        # Log info - ORIGINAL: line 537-540
        log.info(
            f"Creating a reader for the <{split}> split of the <{self._name}> dataset."
        )

        # Get audio directory from manifest - ORIGINAL: line 545-547
        audio_dir = self._retrieve_audio_directory(split)

        # Build manifest reading pipeline - ORIGINAL: line 548
        # OUTPUT: Stream of individual examples: {"audio": "path.wav", "audio_size": 32000}
        builder = self._read_manifest(split, audio_dir, min_audio_len, max_audio_len)

        # Example shuffling - ORIGINAL: line 550-553
        # OUTPUT: Shuffled stream of individual examples (same structure)
        # TODO: (cirquit) how is this different from the actual shuffle window?
        if example_shuffle_window > 1:
            builder.shuffle(example_shuffle_window, seed)
            seed += 1

        # Shard for distributed training - ORIGINAL: line 555-557
        # OUTPUT: Rank-specific stream of individual examples
        if gangs.dp.size > 1:
            builder.shard(gangs.dp.rank, gangs.dp.size, allow_uneven=True)
            seed += gangs.dp.rank

        # Batching - ORIGINAL: line 559-566 (add_bucketing_pipeline call)
        # OUTPUT: Batches of similar-length examples (metadata only)
        # [{
        #   "audio": "audio1.wav", "audio_size": 32000,
        #   "audio": "audio2.wav", "audio_size": 32100,
        #   ...×46 examples
        # }]
        batch_pipeline = BatchingPipeline()
        if batching_strategy == BatchingStrategy.STATIC:
            builder = batch_pipeline.add_static_batching(builder, batch_size, drop_remainder)  # type: ignore
        else:
            builder = batch_pipeline.add_length_batching(
                builder,
                min_audio_len,
                max_audio_len,
                max_num_elements,
                num_seqs_multiple_of,
                drop_remainder,
            )
        # Batch shuffling - ORIGINAL: handled in add_bucketing_pipeline
        builder = batch_pipeline.add_batch_shuffling(
            builder, batch_shuffle_window, seed
        )
        seed += 1  # ORIGINAL: line 567 - seed increment after batching

        # Audio processing pipeline - ORIGINAL: line 568-571
        # Path resolution: Add resolved paths to metadata
        # OUTPUT: [{"audio": {"path": Path("/data/audio1.wav")}, "audio_size": 32000, ...}]
        pipeline = AudioProcessingPipeline()
        builder = pipeline.add_path_resolution(builder, audio_dir, cached_fd_count)

        # Audio decoding: Load actual audio tensors
        # OUTPUT: [{"audio": {"path": Path(...), "data": {"waveform": Tensor[32000]}}, ...}]
        builder = pipeline.add_audio_decoding(builder, dtype, normalize_audio, npc)

        # Audio post-processing: Apply augmentation and feature extraction
        if use_fbank:
            # OUTPUT: [{"audio": {"data": {"fbank": Tensor[80, TIME_FRAMES]}}, ...}]
            builder = pipeline.add_fbank_processing(builder, dtype, npc)
        else:
            # OUTPUT: [{"audio": {"data": {"waveform": Tensor[32000]}}, ...}] (processed)
            builder = pipeline.add_waveform_processing(
                builder,
                normalize_audio,
                dtype,
                spec_aug_p,
                spec_aug_freq_mask_param,
                spec_aug_time_mask_param,
            )

        # Feature renaming: Move features to "audio_feature" key
        # OUTPUT: [{"audio_feature": Tensor[32000], "audio": {...}, ...}]
        builder = pipeline.add_feature_renaming(builder, use_fbank)

        # Audio cropping - ORIGINAL: line 572-574
        # OUTPUT: [{"audio_feature": Tensor[32000]}, ...] (uniform length within batch)
        builder = pipeline.add_audio_cropping(
            builder, seed, max_audio_len, crop_to_batch_minimal_size=no_padding
        )

        # [{'audio': {
        #               'path': 'train-other-500/8272/279781/8272-279781-0011.flac',
        #               'data': {'sample_rate': 16000.0, 'format': 1507330}},
        #  'audio_size': 223120,
        #  'audio_feature': tensor([ 3.0518e-05,  3.6621e-04, 5.1880e-04,  ..., -1.8311e-04, -1.2207e-04, -1.8311e-04],
        # dtype=torch.float16)
        # }, (...) ]

        # Collation - ORIGINAL: line 577-578
        # OUTPUT: {"audio_feature": Tensor[BATCH_SIZE, 32000], ...} (collated tensors)
        collater = Collater(pad_value=None if no_padding else 0)
        builder.map(collater)

        # Return only the first `max_num_batches` - ORIGINAL: line 580-582
        if max_num_batches is not None:
            builder.take(max_num_batches)

        # Prefetch - ORIGINAL: line 584
        builder.prefetch(num_prefetch)

        # Convert to SequenceBatch - ORIGINAL: line 586-588
        # OUTPUT: SequenceBatch(seqs=Tensor[BATCH_SIZE, 32000], padding_mask=None, example={...})
        builder.map(partial(create_sequence_batch, no_padding=no_padding))

        pipeline = builder.and_return()

        # Return DataPipelineReader - ORIGINAL: line 593 return pattern
        return DataPipelineReader[SequenceBatch](
            self._name,
            split,
            pipeline,
            gangs,
            num_accumulate=num_accumulate,
            sync=sync_batches,  # ORIGINAL: options.sync_batches
            sync_mode=sync_mode,  # ORIGINAL: options.sync_mode
        )


@dataclass
class Wav2Vec2TrainDatasetConfig:
    """
    This configuration matches the keys after the top-level `dataset_config:` key
    in the YAML asset definition:

    ```yaml
    name: mydataset
    dataset_config:
      - data: (all keys here must have a companion parameter in this config)
    ```
    """

    data: Path = field(default_factory=Path)


@dataclass(kw_only=True)
class Wav2Vec2TrainDatasetSection(DatasetSection):
    """
    Dataset section for wav2vec2 training configuration.

    ORIGINAL: fairseq2:e9fbd6/src/fairseq2/recipes/wav2vec2/_train.py:117-161
    Class: Wav2Vec2TrainDatasetSection(DatasetSection)
    """

    name: str | None = "librispeech_960h"  # ORIGINAL: line 119
    """The name, path or path to the asset card of the speech dataset."""

    family: str = WAV2VEC2_DATASET  # ORIGINAL: line 122

    path: Path | None = None  # ORIGINAL: line 124
    """The path of the directory with a `.tsv` manifest."""

    train_split: str = "train"  # ORIGINAL: line 126
    """The name of the train data split."""

    valid_split: str | None = "valid"  # ORIGINAL: line 129
    """The name of the valid data split."""

    min_audio_len: int = 32_000  # ORIGINAL: line 132
    """The minimum audio sequence length."""

    max_audio_len: int = 250_000  # ORIGINAL: line 135
    """The maximum audio sequence length."""

    # Batching configuration - NEW: replaces batching parameter
    batching_strategy: BatchingStrategy = BatchingStrategy.LENGTH
    """Batching strategy is defined through an enum:
    - BatchingStrategy.LENGTH ("length") = Specifies batching where each batch has a maximum number of elements.
    - BatchingStrategy.STATIC ("static") = Specifies batching where each batch has the same number of examples.
    """

    batch_size: int | None = None
    """If `batching_strategy = BatchingStrategy.STATIC`, ignores `max_num_tokens` and each batch will have `batch_size` examples.
    """

    num_seqs_multiple_of: int = 8
    """If `batching_strategy = BatchingStrategy.LENGTH, ignores `batch_size` and each batch will have
    `<= max_num_elements` elements with `<= num_seqs_multipe_of` or `num_seqs_multiple_of * N` sequences.
    This is primarily for hardware optimization, but will only work when sufficient enough sequences are available for bucketing.
    """

    max_num_elements: int = 1_500_000  # ORIGINAL: line 138
    """If `batching_strategy = BatchingStrategy.LENGTH, ignores `batch_size` and each batch will have
    `<= max_num_elements` elements with `<= num_seqs_multipe_of` or `num_seqs_multiple_of * N` sequences.
    This is primarily for hardware optimization, but will only work when sufficient enough sequences are available for bucketing.
    """

    normalize_audio: bool = False  # ORIGINAL: line 141
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    use_fbank: bool = False  # ORIGINAL: SpeechReadOptions line 185
    """If ``True``, use fbank features instead of waveform."""

    no_padding: bool = True  # ORIGINAL: SpeechReadOptions line 188
    """If ``True``, all elements in the batch will be truncated to by batch minimal length.
    Therefore, no padding will be applied to the batch.
    """

    npc: int = 10  # ORIGINAL: SpeechReadOptions line 193
    """The number of parallel calls to use in the pipeline."""

    # Upsampling - ORIGINAL: SpeechReadOptions lines 197-199
    beta_corpus: float | None = None
    """Corpus sampling temperature; between [0,1]."""

    beta_language: float | None = None
    """Language sampling temperature; between [0,1]."""

    # SpecAugment - ORIGINAL: SpeechReadOptions lines 202-207
    spec_aug_p: float | None = None  # ORIGINAL: line 202
    """Probability of applying SpecAugment per row."""

    spec_aug_freq_mask_param: int = 80  # ORIGINAL: line 204
    """Maximum frequency mask length."""

    spec_aug_time_mask_param: int = 80  # ORIGINAL: line 206
    """Maximum time mask length."""

    # Shuffling - ORIGINAL: DataReadOptions lines 53-65
    example_shuffle_window: int = (
        500_000  # ORIGINAL: DataReadOptions line 53 (overridden in v0.4 config)
    )
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = (
        0  # ORIGINAL: DataReadOptions line 60 (overridden in v0.4 config)
    )
    """The size of the sliding window for shuffling batches. Zero equals full dataset shuffling."""

    # Batching behavior - ORIGINAL: DataReadOptions lines 67-86
    drop_remainder: bool = False  # ORIGINAL: DataReadOptions line 67
    """If ``True``, drops the last set of batches if they have in total fewer examples than requested."""

    sync_batches: bool = True  # ORIGINAL: DataReadOptions line 73
    """If ``True``, ensures that each process reads the same number of batches."""

    sync_mode: SyncMode = SyncMode.UNTIL_FIRST  # ORIGINAL: DataReadOptions line 82
    """The data synchronization mode among processes."""

    # Performance - ORIGINAL: DataReadOptions lines 88-104
    max_num_batches: int | None = None  # ORIGINAL: DataReadOptions line 88
    """The maximum number of batches to return."""

    num_accumulate: int = 1  # ORIGINAL: DataReadOptions line 91
    """The number of batches to accumulate in each iteration."""

    num_prefetch: int = (
        4  # ORIGINAL: DataReadOptions line 97 (overridden in v0.4 config)
    )
    """The number of batches to prefetch in background."""

    seed: int = 2  # ORIGINAL: DataReadOptions line 100
    """The seed to initialize the random number generators used internally."""

    cached_fd_count: int = 1000  # NEW: Hidden in FileMapper configs
    """Enables an LRU cache on the last ``cached_fd_count`` files read.
    ``FileMapper`` will memory map all the cached file, so this is especially
    useful for reading several slices of the same file.
    """


def open_wav2vec2_train_dataset(
    name: str, config: Wav2Vec2TrainDatasetConfig
) -> Wav2Vec2TrainDataset:
    """The mapping between the dataset asset card definition and the Wav2Vec2TrainDataset."""
    return Wav2Vec2TrainDataset.from_path(config.data, name)
