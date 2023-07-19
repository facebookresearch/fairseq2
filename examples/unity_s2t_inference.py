# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from pathlib import Path

import torch

from fairseq2.data import Collater, DataPipeline, FileMapper
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.models.unity import load_unity_s2t_model, load_unity_text_tokenizer
from fairseq2.models.unity.translator import Translator
from fairseq2.typing import Device


@dataclass
class InferenceContext:
    model_name: str
    """The name of the S2T UnitY model."""

    data_file: Path
    """The pathname of the test TSV data file."""

    audio_root_dir: Path
    """The pathname of the directory under which audio files are stored."""

    target_lang: str
    """The target translation language."""

    batch_size: int
    """The batch size for model input."""

    device: Device
    """The device on which to run inference."""


def build_data_pipeline(ctx: InferenceContext) -> DataPipeline:
    # TODO: This will be soon auto-tuned. Right now hand-tuned for devfair.
    n_parallel = 4

    # Open TSV, skip the header line, split into fields, and return three fields
    # only.
    split_tsv = StrSplitter(
        names=["id", "audio_file", "raw_target_text"], indices=[0, 1, 3]
    )

    pipeline_builder = read_text(ctx.data_file, rtrim=True).skip(1).map(split_tsv)

    # Memory map audio files and cache up to 10 files.
    map_file = FileMapper(root_dir=ctx.audio_root_dir, cached_fd_count=10)

    pipeline_builder.map(map_file, selector="audio_file", num_parallel_calls=n_parallel)

    # Decode mmap'ed audio using libsndfile and convert them from waveform to
    # fbank.
    decode_audio = AudioDecoder(dtype=torch.float32)

    convert_to_fbank = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=True,
        device=ctx.device,
        dtype=torch.float16,
    )

    pipeline_builder.map(
        [decode_audio, convert_to_fbank],
        selector="audio_file.data",
        num_parallel_calls=n_parallel,
    )

    # Batch every 4 line
    pipeline_builder.bucket(bucket_size=ctx.batch_size)

    collate = Collater(pad_idx=0, pad_to_multiple=2)

    pipeline_builder.map(collate, num_parallel_calls=n_parallel)

    # Prefetch up to 4 batches in background.
    pipeline_builder.prefetch(4)

    # Build and return the data pipeline.
    return pipeline_builder.and_return()


def run_inference(ctx: InferenceContext) -> None:
    # Load the demo S2T Unity model in fp16.
    model = load_unity_s2t_model(ctx.model_name, device=ctx.device, dtype=torch.float16)

    model.eval()

    # Load the tokenizer. As of today, it is NLLB-200 or NLLB-100 depending on
    # the model.
    tokenizer = load_unity_text_tokenizer(ctx.model_name)

    # Build a simple pipeline that just reads a single TSV file.
    pipeline = build_data_pipeline(ctx)

    # TODO: This is a temporary class with potential perf improvements and will
    # be revised soon.
    translator = Translator(model, tokenizer, ctx.target_lang, ctx.device)

    # Iterate through each example in the TSV file until CTRL-C.
    for example in pipeline:
        speech = example["audio_file"]["data"]["fbank"]

        translation = translator.translate_batch(speech["seqs"], speech["seq_lens"])

        reference_text = example["raw_target_text"]

        for i in range(ctx.batch_size):
            print(f"Ref: {reference_text[i]}")
            print(f"Out: {translation[i]}")
            print()


if __name__ == "__main__":
    # fmt: off
    ctx = InferenceContext(
        model_name="unity_s2t_demo",
        data_file=Path("/large_experiments/seamless/ust/balioglu/sample-datasets/test_cvst2_spa-eng.tsv"),
        audio_root_dir=Path("/large_experiments/seamless/ust/data/audio_zips"),
        target_lang="eng_Latn",
        batch_size=4,
        device=torch.device("cuda:0"),
    )
    # fmt: on

    run_inference(ctx)
