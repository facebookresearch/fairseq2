# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, TextIO, Tuple, Union, final

import torch

from fairseq2.assets import AssetNotFoundError, default_asset_store
from fairseq2.assets.utils import retrieve_asset_card
from fairseq2.checkpoint import CheckpointModelMetadataProvider
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import TextTokenizer, load_text_tokenizer
from fairseq2.datasets import StaticBatching
from fairseq2.datasets.text import GenericTextDataset, load_text_dataset
from fairseq2.gang import Gang
from fairseq2.generation import (
    BeamSearchSeq2SeqGenerator,
    Sampler,
    SamplingSeq2SeqGenerator,
    Seq2SeqGenerator,
    StandardBeamSearchAlgorithm,
    TopKSampler,
    TopPSampler,
)
from fairseq2.generation.text import SequenceToTextConverter
from fairseq2.logging import get_log_writer
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.transformer import load_transformer_model
from fairseq2.recipes.common_metrics import Seq2SeqGenerationMetricBag
from fairseq2.recipes.generator import AbstractGeneratorUnit, Generator
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import broadcast_model, setup_root_gang
from fairseq2.typing import META, DataType, override
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass
class TextTranslateConfig:
    """Holds the configuration of a text translation task."""

    # Data
    dataset: Union[str, Path] = "foo"  # TODO: change!
    """The name, path, or path to the asset card of the text dataset."""

    source_lang: str = "eng_Latn"
    """The code of the language to translate from."""

    target_lang: str = "deu_Latn"
    """The code of the language to translate to."""

    max_seq_len: int = 512
    """The maximum sequence length."""

    batch_size: int = 1
    """The input batch size."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    # Model
    model: Union[str, Path] = "nllb-200_dense_distill_600m"
    """The name of the model to translate with."""

    checkpoint_dir: Optional[Path] = None
    """The checkpoint directory containing models saved by :class:`FileCheckpointManager`."""

    dtype: DataType = torch.float16
    """The data type of the model."""

    # Generation
    generator_mode: Literal["beam_search", "sampling"] = "beam_search"
    """The mode of sequence generation."""

    beam_search: BeamSearchConfig = field(default_factory=lambda: BeamSearchConfig())
    """The configuration for beam search-based sequence generation."""

    sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig())
    """The configuration for sampling-based sequence generation."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""


@dataclass
class BeamSearchConfig:
    """Holds the configuration for beam search-based sequence generation.

    See :class:`BeamSearchSeq2SeqGenerator` for more info.
    """

    algorithm: Literal["standard"] = "standard"
    """The beam search algorithm."""

    beam_size: int = 5
    """The beam size."""

    min_gen_len: int = 1
    """The minimum generation length."""

    max_gen_len: Tuple[int, int] = (1, 128)
    """The maximum generation length."""

    max_seq_len: Optional[int] = None
    """The maximum sequence length including prompt."""

    normalize_scores: bool = True
    """If ``True``, normalizes scores by lengths of generated sequences."""

    temperature: float = 1.0
    """The logit temperature."""

    unk_penalty: float = 0.0
    """The UNK symbol penalty."""

    len_penalty: float = 1.0
    """The length penalty."""

    prefill_chunk_size: Optional[int] = 512
    """The prefill will be performed incrementally by chunks of this size."""

    decode_capacity_increment: Optional[int] = 16
    """The sequence length capacity will be incremented by multiplies of this value."""


@dataclass
class SamplingConfig:
    """Holds the configuration for sampling-based sequence generation.

    See :class:`SamplingSeq2SeqGenerator` for more info.
    """

    sampler: Literal["top-p", "top-k"] = "top-p"
    """The sampling algorithm."""

    top_p: float = 0.9
    """The cumulative probability threshold for top-p sampling."""

    top_k = 10
    """The number of top candidates to select from for top-k sampling."""

    min_gen_len: int = 1
    """The minimum generation length."""

    max_gen_len: Tuple[int, int] = (1, 128)
    """The maximum generation length."""

    max_seq_len: Optional[int] = None
    """The maximum sequence length including prompt."""

    compute_scores: bool = False
    """If ``True``, computes scores of generated sequences."""

    normalize_scores: bool = True
    """If ``True``, normalizes scores by lengths of generated sequences."""

    temperature: float = 0.6
    """The logit temperature."""

    unk_penalty: float = 0.0
    """The UNK symbol penalty."""

    len_penalty: float = 1.0
    """The length penalty."""

    prefill_chunk_size: Optional[int] = 512
    """The prefill will be performed incrementally by chunks of this size."""

    decode_capacity_increment: Optional[int] = 16
    """The sequence length capacity will be incremented by multiplies of this value."""


text_translate_presets = ConfigRegistry[TextTranslateConfig]()

text_translate_preset = text_translate_presets.decorator


@text_translate_preset("nllb_dense_600m")
def _nllb_dense_600m() -> TextTranslateConfig:
    return TextTranslateConfig()


def load_text_translator(
    config: TextTranslateConfig, output_dir: Path
) -> Generator[SequenceBatch]:
    """Load a :class:`Generator` for text translation."""
    wall_watch = Stopwatch(start=True)

    if config.checkpoint_dir is not None:
        default_asset_store.metadata_providers.append(
            CheckpointModelMetadataProvider(config.checkpoint_dir)
        )

    gang = setup_root_gang(log)

    seed = config.seed

    # Load the tokenizer.
    model_card = retrieve_asset_card(config.model)

    log.info("Loading {} tokenizer.", model_card.name)

    tokenizer = load_text_tokenizer(model_card)

    log.info("Tokenizer loaded.")

    # Load the dataset.
    try:
        dataset_card = retrieve_asset_card(config.dataset)
    except AssetNotFoundError:
        dataset_card = None

    if dataset_card is not None:
        log.info("Loading {} text dataset.", dataset_card.name)

        dataset = load_text_dataset(dataset_card)

        log.info("Dataset loaded.")
    else:
        try:
            path = Path(config.dataset)
        except ValueError:
            raise AssetNotFoundError(
                config.dataset, f"An asset with the name '{config.dataset}' cannot be found."  # type: ignore[arg-type]
            )

        dataset = GenericTextDataset.from_path(path)

    # Load the model.
    log.info("Loading {} model on rank 0.", model_card.name)

    if gang.rank == 0:
        init_device = gang.device
    else:
        init_device = META

    model = load_transformer_model(model_card, device=init_device, dtype=config.dtype)

    gang.barrier()

    log.info("Model loaded on rank 0.")

    # Distribute the model to all processes in the gang.
    if gang.size != 1:
        broadcast_model(model, gang, log)

    log_model(model, log)

    # Initialize the sequence generator.
    generator = _create_sequence_generator(
        model, config.generator_mode, config.beam_search, config.sampling
    )

    # Initialize the generator unit.
    text_output_file = output_dir.joinpath(
        f"translations/{config.source_lang}-{config.target_lang}/rank_{gang.rank}.txt"
    )

    json_output_file = output_dir.joinpath(
        f"translations/{config.source_lang}-{config.target_lang}/rank_{gang.rank}.jsonl"
    )

    try:
        text_output_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        raise RuntimeError(
            f"The output directory '{text_output_file.parent}' cannot be created. See nested exception for details."
        ) from ex

    try:
        text_output_fp = text_output_file.open("w")
    except OSError as ex:
        raise RuntimeError(
            f"The output file '{text_output_file}' cannot be created. See nested exception for details."
        ) from ex

    try:
        json_output_fp = json_output_file.open("w")
    except OSError as ex:
        raise RuntimeError(
            f"The output file '{json_output_file}' cannot be created. See nested exception for details."
        ) from ex

    unit = TextTranslationUnit(
        generator, tokenizer, config.target_lang, gang, text_output_fp, json_output_fp
    )

    text_encoder = tokenizer.create_encoder(
        task="translation", lang=config.source_lang, mode="source"
    )

    data_reader = dataset.create_reader(
        text_encoder,
        tokenizer.vocab_info.pad_idx,
        gang,
        config.max_seq_len,
        batching=StaticBatching(config.batch_size),
        num_prefetch=config.num_prefetch,
        seed=seed,
    )

    seed += 1

    # Initialize the generator.
    return Generator[SequenceBatch](
        unit=unit,
        data_reader=data_reader,
        root_gang=gang,
        metrics_dir=output_dir.joinpath("metrics"),
        seed=seed,
        wall_watch=wall_watch,
    )


@final
class TextTranslationUnit(AbstractGeneratorUnit[SequenceBatch]):
    """Represents a text translation unit."""

    _converter: SequenceToTextConverter
    _text_output_stream: TextIO
    _json_output_stream: TextIO
    _metric_bag: Seq2SeqGenerationMetricBag

    def __init__(
        self,
        generator: Seq2SeqGenerator,
        tokenizer: TextTokenizer,
        target_lang: str,
        gang: Gang,
        text_output_stream: TextIO,
        json_output_stream: TextIO,
    ) -> None:
        super().__init__(generator.model)

        self._converter = SequenceToTextConverter(
            generator, tokenizer, task="translation", target_lang=target_lang
        )

        self._text_output_stream = text_output_stream
        self._json_output_stream = json_output_stream

        self._metric_bag = Seq2SeqGenerationMetricBag(gang)

    @override
    def __call__(self, batch: SequenceBatch) -> None:
        if batch.example is None:
            raise ValueError("`batch.example` must not be `None`.")

        try:
            refs = batch.example["text"]
        except KeyError:
            raise ValueError("`batch.example` must contain a 'text' item.")

        hyps, output = self._converter.batch_convert(batch.seqs, batch.padding_mask)

        self._metric_bag.update_batch_metrics(output, batch.num_elements())

        # Dump as text.
        stream = self._text_output_stream

        for ref, hyp in zip(refs, hyps):
            stream.write("REF: ")
            stream.write(ref)
            stream.write("\n")
            stream.write("HYP: ")
            stream.write(hyp)
            stream.write("\n\n")

        stream.flush()

        # Dump as JSON.
        stream = self._json_output_stream

        for ref, hyp in zip(refs, hyps):
            json.dump({"ref": ref, "hyp": hyp}, stream, indent=None)

            stream.write("\n")

        stream.flush()

    @property
    @override
    def metric_bag(self) -> Seq2SeqGenerationMetricBag:
        return self._metric_bag


def _create_sequence_generator(
    model: EncoderDecoderModel,
    mode: str,
    beam_search_config: BeamSearchConfig,
    sampling_config: SamplingConfig,
) -> Seq2SeqGenerator:
    if mode == "beam_search":
        return _create_beam_search_generator(model, beam_search_config)

    if mode == "sampling":
        return _create_sampling_generator(model, sampling_config)

    raise ValueError(
        f"`config.generator_mode` must be 'sampling' or 'beam_search', but is '{mode}' instead."
    )


def _create_beam_search_generator(
    model: EncoderDecoderModel, config: BeamSearchConfig
) -> BeamSearchSeq2SeqGenerator:
    if config.algorithm == "standard":
        algorithm = StandardBeamSearchAlgorithm()
    else:
        raise ValueError(
            f"`config.beam_search.algorithm` must be 'standard', but is '{config.algorithm}' instead."
        )

    return BeamSearchSeq2SeqGenerator(
        model,
        algorithm=algorithm,
        beam_size=config.beam_size,
        min_gen_len=config.min_gen_len,
        max_gen_len=config.max_gen_len,
        echo_prompt=True,
        normalize_scores=config.normalize_scores,
        temperature=config.temperature,
        unk_penalty=config.unk_penalty,
        len_penalty=config.len_penalty,
        prefill_chunk_size=config.prefill_chunk_size,
        decode_capacity_increment=config.decode_capacity_increment,
    )


def _create_sampling_generator(
    model: EncoderDecoderModel, config: SamplingConfig
) -> SamplingSeq2SeqGenerator:
    sampler: Sampler

    if config.sampler == "top-p":
        sampler = TopPSampler(config.top_p)
    elif config.sampler == "top-k":
        sampler = TopKSampler(config.top_k)
    else:
        raise ValueError(
            f"`config.sampling.sampler` must be 'top-p' or 'top-k', but is '{config.sampler}' instead."
        )

    return SamplingSeq2SeqGenerator(
        model,
        sampler,
        min_gen_len=config.min_gen_len,
        max_gen_len=config.max_gen_len,
        max_seq_len=config.max_seq_len,
        echo_prompt=True,
        compute_scores=config.compute_scores,
        normalize_scores=config.normalize_scores,
        temperature=config.temperature,
        unk_penalty=config.unk_penalty,
        len_penalty=config.len_penalty,
        prefill_chunk_size=config.prefill_chunk_size,
        decode_capacity_increment=config.decode_capacity_increment,
    )
