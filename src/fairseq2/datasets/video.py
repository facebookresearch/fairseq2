# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Final, Sequence, Tuple, cast, final

import torch

from fairseq2.data import read_sequence, Collater, DataPipeline, DataPipelineBuilder
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.data.text.tokenizers.tokenizer import TextTokenizer
from fairseq2.datasets.data_reader import DataPipelineReader, DataReader
from fairseq2.datasets.error import DatasetError, SplitNotFoundError
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gang
from fairseq2.data.video import VideoDecoder, VideoSegmentationConfig, NoSegmentation, LengthBasedSegmentation, CountBasedSegmentation, segment_video_by_count, segment_video_by_length
from fairseq2.datasets.config import Batching, DataReadOptions, StaticBatching
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.nn.padding import pad_seqs
from fairseq2.typing import DataType


# TODO: FIX, INFER
npc = 10


GENERIC_TEXT_VIDEO_DATASET_FAMILY: Final = "generic_text_video"



@dataclass(kw_only=True)
class ManifestReadOptions:
    
    video_column: int
    """Index of the column that contains video paths in the manifest file"""
    
    text_column: int | None = None
    """Index of the column that contains text content in the manifest file (Optional)"""


@dataclass
class VideoReadOptions(DataReadOptions):
    dtype: DataType = torch.float32
    """The data type of the decoded audio sequences."""

    frame_step: int = 4
    """Step size to sample the video frames."""

    segmenation: VideoSegmentationConfig = field(
        default_factory=lambda: NoSegmentation()
    )


class VideoTextDataset(ABC):
    """Represents a video dataset as source with a text as target"""

    @abstractmethod
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        batching: Batching,
        min_video_size: int = 1024,
        max_video_size: int | None = None,
        manifest_options: ManifestReadOptions | None = None,
        video_options: VideoReadOptions | None = None,
    ) -> DataReader[Seq2SeqBatch]:
        """Create a dataset reader.

        :param gang:
            The gang over which to shard the dataset.
        :param min_video_size:
            Skip the videos that are too short (smaller than `min_video_size` bytes)
        :param max_video_size:
            If specified, all videos that exceed `max_video_size` bytes will be skipped
        """


@final
class GenericVideoTextDataset(VideoTextDataset):
    
    _name: str
    _manifest_dir: Path
    _splits: set[str]

    def __init__(self, name: str, manifest_dir: Path, splits: set[str]) -> None:
        """
        :param manifest_dir:
            The directory under which the manifest files resides.
        :param splits:
            The available splits.
        """
        self._name = name
        self._manifest_dir = manifest_dir
        self._splits = splits

    @staticmethod
    def from_path(path: Path, name: str | None = None) -> GenericVideoTextDataset:
        if name is None:
            name = f"path:{path.name}"

        path = path.expanduser().resolve()

        if not path.is_dir():
            return GenericVideoTextDataset(name, manifest_dir=path.parent, splits={path.stem})

        try:
            splits = {f.stem for f in path.glob("*.tsv")}
        except OSError as ex:
            raise DatasetError(
                f"The splits under the '{path}' directory cannot be determined. See the nested exception for details."
            ) from ex

        return GenericVideoTextDataset(name, path, splits) 

    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        batching: Batching,
        min_video_size: int = 1024,
        max_video_size: int | None = None,
        manifest_options: ManifestReadOptions | None = None,
        video_options: VideoReadOptions | None = None,
    ) -> DataReader[Seq2SeqBatch]:
        
        if split not in self._splits:
            raise SplitNotFoundError(self._name, split, self._splits)
        
        if video_options is None:
            video_options = VideoReadOptions()
            
        seed = video_options.seed
        
        if manifest_options is None:
            manifest_options = ManifestReadOptions()
        
        builder = self._read_manifest(split, options=manifest_options)
        
        # TODO: handle shuffle

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank
        
        # Segment video
        if isinstance(video_options.segmenation, LengthBasedSegmentation):
            segment_func = partial(
                segment_video_by_length,
                frame_step=video_options.frame_step,
                num_frames_in_segment=video_options.segmenation.num_frames_in_segment,
            ) 
        elif isinstance(video_options.segmenation, CountBasedSegmentation):
            segment_func = partial(
                segment_video_by_count,
                frame_step=video_options.frame_step,
                num_segments=video_options.segmenation.num_segments,
            )
        else:
            segment_func = lambda _: None
            
        def segment_video(video: Path) -> dict[str, Any]:
            segments = segment_func(video)
            return {"file_path": video, "segments": segments}
        
        builder.map(segment_video, selector="video")
        
        # Read video
        video_decoder = VideoDecoder(
            dtype=video_options.dtype,
            min_video_size=min_video_size,
            max_video_size=max_video_size,
        )

        def decode_video(video: dict[str, Any]) -> Sequence[torch.Tensor] | None:
            return video_decoder(**video)

        builder.map(decode_video, selector="video")

        # Filter the empty segments and flatten
        builder.filter(lambda example: example["video"] is not None)
        
        # Read text and tokenize
        text_encoder = tokenizer.create_encoder()

        builder.map(text_encoder, selector="text", num_parallel_calls=npc)

        def flatten_segments(example: dict[str, Any]) -> DataPipeline:
            segments = example["video"]
            segments = [(segment, example["text"]) for segment in segments]
            return read_sequence(segments).and_return()

        builder.yield_from(flatten_segments)

        # bucket with size
        if not isinstance(batching, StaticBatching):
            raise NotSupportedError(f"`{batching} is not supported.")
        
        builder.bucket(batching.batch_size, drop_remainder=video_options.drop_remainder)
        
        # Shuffle buckets.
        if video_options.batch_shuffle_window != 1:
            builder.shuffle(video_options.batch_shuffle_window, seed)
            
            
        # Collate the bucketed examples
        collater = Collater(pad_value=0)
        builder.map(collater)
        
        # Convert to Seq2Seq
        def to_batch(example: Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]) -> Seq2SeqBatch:
            source_seqs, source_padding_mask = pad_seqs(example[0])
            target_seqs, target_padding_mask = pad_seqs(example[1])
            
            return Seq2SeqBatch(
                source_seqs,
                source_padding_mask,
                target_seqs,
                target_padding_mask,
                example,
            )
        
        pipeline = builder.map(to_batch).and_return()
        
        return DataPipelineReader[Seq2SeqBatch](
            self._name,
            pipeline,
            gang,
            num_accumulate=video_options.num_accumulate,
            drop_remainder=video_options.drop_remainder,
            sync_batches=video_options.sync_batches,
            sync_mode=video_options.sync_mode,
        )

    def _read_manifest(self, split: str, options: ManifestReadOptions) -> DataPipelineBuilder:

        def read_tsv_file() -> DataPipelineBuilder:
            tsv_file = self._manifest_dir.joinpath(f"{split}.tsv")

            builder = read_text(tsv_file, rtrim=True, memory_map=True)

            builder.skip(1)  # Path to the data directory.

            video_col, text_col = options.video_column, options.text_column
            
            if text_col is None:
                raise ValueError("Text column is required")

            field_splitter = StrSplitter(names=["video", "text"], indices=[video_col, text_col])

            builder.map(field_splitter, num_parallel_calls=npc)

            return builder

        manifest = list(read_tsv_file().and_return())

        return read_sequence(manifest)
