# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    DataPipelineError,
    FileMapper,
)
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.data.text import StrSplitter, StrToTensorConverter, read_text
from fairseq2.typing import Device


@dataclass
class SpeechToTextDatasetConfig:
    root_dir: Path
    """The root directory of the dataset."""

    audio_root_dir: Path
    """The root directory of the zipped audio files."""

    split: str
    """The split to read."""

    max_num_tokens: int = 4096
    """The maximum number of unit tokens per batch."""

    unit_pad_idx: int = 0
    """The pad index to use in bucketing."""

    num_examples_to_read: Optional[int] = None
    """The number of examples (i.e. batches) to read from the dataset."""

    prefetch: int = 4
    """The number of batches to prefetch in a background thread."""

    rank: int = 0
    """The rank of this worker in the process group."""

    world_size: int = 1
    """The world size of the process group."""

    num_parallel_calls: int = 4
    """The number of parallel calls in map operations."""

    shuffle_window: int = 1000
    """The sliding shuffle window."""

    device: Device = Device("cpu")
    """The device on which to allocate batches."""


def create_file_pipeline(
    pathname: Path, config: SpeechToTextDatasetConfig
) -> DataPipeline:
    """Creates a data pipeline for the specified TSV file."""

    # Memory map `pathname` and read it in text mode (skip empty lines if any).
    pipeline_builder = read_text(pathname, rtrim=True, skip_empty=True, memory_map=True)

    # Skip the header line.
    tsv_lines = pipeline_builder.skip(1).and_return()

    # We use the pseudo `line_numbers` and `filename` pipelines for
    # troubleshooting purposes. If reading an example fails, they gives us the
    # exact position to diagnose the issue.
    line_numbers = DataPipeline.count().and_return()

    filename = DataPipeline.constant(pathname).and_return()

    # Effectively prepend line number and filename to each line read from the
    # TSV file.
    pipeline_builder = DataPipeline.zip(
        [line_numbers, filename, tsv_lines],
        names=["line_number", "filename", "line"],
        zip_to_shortest=True,
    )

    # Read every `world_size`th line starting from `rank`th item in the file.
    pipeline_builder.shard(config.rank, config.world_size)

    # And, create the pipeline for the TSV file.
    return pipeline_builder.and_return()


def create_s2t_pipeline(config: SpeechToTextDatasetConfig) -> DataPipeline:
    """Creates the Speech-to-Text data pipeline."""

    # For demo purposes, list all TSV files at the root data directory with the
    # English target direction.
    tsv_files = config.root_dir.glob(f"{config.split}_cv12_*-eng.tsv")

    # For each TSV file, create a "sub-pipeline" in `create_file_pipeline`.
    tsv_pipelines = [create_file_pipeline(f, config) for f in tsv_files]

    # Now, pick the next line to read in round-robin fashion from file
    # pipelines. In a real-world use case, instead of round-robin, we would use
    # the `sample()` operator that takes a weight for each sub-pipeline.
    pipeline_builder = DataPipeline.round_robin(tsv_pipelines)

    # Shuffle the lines with a sliding window of 1000 items. Note in real-world
    # typically the window should be at least an order magnitude larger.
    pipeline_builder.shuffle(config.shuffle_window, enabled=config.split == "train")

    # Split each text line into its fields. For demo purposes "exclude" (i.e.
    # ignore) `tgt_lang`, `src_lang`.
    fields = ["id", "audio_file", "num_frames", "tgt_text", "units"]

    txt_splitter = StrSplitter(names=fields, indices=[5, 6], exclude=True)

    pipeline_builder.map(txt_splitter, selector="line")

    # Memory map each `audio_file` using an LRU cache of size 100 for file
    # descriptors.
    map_file = FileMapper(config.audio_root_dir, cached_fd_count=100)

    pipeline_builder.map(
        map_file,
        selector="line.audio_file",
        num_parallel_calls=config.num_parallel_calls,
    )

    # Decode each mmap'ed audio file using libsndfile.
    decode_audio = AudioDecoder()

    # And, convert from waveform to log-mel filterbank in fp16.
    convert_to_fbank = WaveformToFbankConverter(
        num_mel_bins=80,
        channel_last=True,
        standardize=True,
        device=config.device,
        dtype=torch.float16,
    )

    pipeline_builder.map(
        [decode_audio, convert_to_fbank],
        selector="line.audio_file.data",
        num_parallel_calls=config.num_parallel_calls,
    )

    # Convert the `n_frames` field into an integer. Note our C++ API knows about
    # basic Python "factory" functions and executes them natively.
    pipeline_builder.map(int, selector="line.num_frames")

    # Convert the `tgt_text` field into a unit tensor.
    # TODO: Allow specifying device.
    convert_unit_tensor = StrToTensorConverter(dtype=torch.int16)

    pipeline_builder.map(
        convert_unit_tensor,
        selector="line.units",
        num_parallel_calls=config.num_parallel_calls,
    )

    # Bucket elements based on the length of their unit tensors.
    bucket_sizes = create_bucket_sizes(max_num_tokens=config.max_num_tokens)

    pipeline_builder.bucket_by_length(
        bucket_sizes,
        selector="line.units",
        # Note that we can also use multiple selectors for length criteria.
        # selector="audio_file.data.fbank,units",
    )

    # Collate bucketed elements into a batch.
    collater = Collater(
        pad_idx=0,
        # Typically, multiplies of 2 or 8 significantly speed up fp16 matmul.
        pad_to_multiple=2,
        # Unlike fbanks, use the unit pad index for the unit tensors.
        overrides=[
            CollateOptionsOverride(
                selector="line.units",
                pad_idx=config.unit_pad_idx,
                pad_to_multiple=2,
            )
        ],
    )

    pipeline_builder.map(collater)

    # For demo purposes, just read 10 items from the pipeline and stop.
    if config.num_examples_to_read:
        pipeline_builder.take(config.num_examples_to_read)

    # Keep reading up to 4 batches in a background thread, while the foreground
    # thread is busy with the training loop.
    pipeline_builder.prefetch(config.prefetch)

    # And, finally build our pipeline. Ignore up to 10 read failures before
    # erroring out. TBD: soon errors will be logged in Python.
    return pipeline_builder.and_return(max_num_warnings=10)


# This will very likely become a utility function in fairseq2.
def create_bucket_sizes(max_num_tokens: int) -> List[Tuple[int, int]]:
    def compute_batch_size(seq_len: int) -> int:
        batch_size = max_num_tokens // seq_len

        # Make sure that our batch sizes are always a multiple of 8.
        return max(8, batch_size - batch_size % 8)

    bucket_sizes = []

    # Create a bucket for each sequence length up to 512,
    for seq_len in range(2, 512):
        bucket_sizes.append((compute_batch_size(seq_len), seq_len))

    # For longer sequence lengths, use 4 as the bucket granularity.
    for seq_len in range(512, max_num_tokens, 4):
        bucket_sizes.append((compute_batch_size(seq_len), seq_len))

    return bucket_sizes


def run_training_loop() -> None:
    # fmt: off
    config = SpeechToTextDatasetConfig(
        root_dir=Path("/large_experiments/seamless/ust/spopuri/H1_2023/S2ST/T2U_V4/manifests_10K"),
        audio_root_dir=Path("/large_experiments/seamless/ust/data/audio_zips"),
        split="train",
        num_examples_to_read=1000,
    )
    # fmt: on

    unity_pipeline = create_s2t_pipeline(config)

    # Iterate through each batch in the pipeline.
    try:
        for batch in unity_pipeline:
            # This is your training/eval/inference code.
            pass
    except DataPipelineError:
        raise

        # If the pipeline is not broken, we can call `next()` on the iterator
        # and get the next example. We do not demonstrate it here.
        # if unity_pipeline.is_broken:
        #    return

    # Once we reach the end of data, we have to "reset" the data pipeline if we
    # want to read it again.
    unity_pipeline.reset()


if __name__ == "__main__":
    run_training_loop()
