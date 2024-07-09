# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, TextIO, Union, final

import torch
from torch.nn import Module

from fairseq2.assets import AssetNotFoundError, default_asset_store
from fairseq2.assets.utils import retrieve_asset_card
from fairseq2.checkpoint import CheckpointModelMetadataProvider
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data.text import TextTokenizer, load_text_tokenizer
from fairseq2.datasets import LengthBatching, StaticBatching
from fairseq2.datasets.parallel_text import (
    Direction,
    GenericParallelTextDataset,
    load_parallel_text_dataset,
)
from fairseq2.gang import Gang
from fairseq2.generation import Seq2SeqGenerator
from fairseq2.generation.text import SequenceToTextConverter
from fairseq2.logging import get_log_writer
from fairseq2.metrics.bleu import BleuMetric
from fairseq2.models.seq2seq import Seq2SeqBatch, as_auto_regressive_input
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.models.transformer import load_transformer_model
from fairseq2.recipes.common_metrics import Seq2SeqGenerationMetricBag, Seq2SeqMetricBag
from fairseq2.recipes.evaluator import AbstractEvalUnit, Evaluator, EvalUnit
from fairseq2.recipes.transformer.translate import (
    BeamSearchConfig,
    SamplingConfig,
    _create_sequence_generator,
)
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import broadcast_model, setup_root_gang
from fairseq2.typing import META, DataType, override
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass
class TransformerEvalConfig:
    """Holds the configuration of a Transformer model evaluation task."""

    # Data
    dataset: Union[str, Path] = "foo"  # TODO: change!
    """The name, path, or path to the asset card of the parallel text dataset."""

    split: str = "test"
    """The name of the test data split."""

    max_seq_len: int = 512
    """The maximum sequence length."""

    max_num_tokens: int = 4096
    """The maximum number of tokens per batch."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    # Model
    model: Union[str, Path] = "nllb-200_dense_distill_600m"
    """The name of the model to evaluate."""

    checkpoint_dir: Optional[Path] = None
    """The checkpoint directory containing models saved by :class:`FileCheckpointManager`."""

    dtype: DataType = torch.float16
    """The data type of the model."""

    # Loss
    label_smoothing: float = 0.1
    """The amount of label smoothing to apply while computing the loss."""

    # Generation
    generator_mode: Literal["beam_search", "sampling"] = "beam_search"
    """The mode of sequence generation."""

    beam_search: BeamSearchConfig = field(default_factory=lambda: BeamSearchConfig())
    """The configuration for beam search-based sequence generation."""

    sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig())
    """The configuration for sampling-based sequence generation."""

    generator_batch_size: int = 1
    """The input batch size to the sequence generator."""

    generator_max_num_batches: Optional[int] = None
    """The maximum number of batches to feed to the sequence generator."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""


transformer_eval_presets = ConfigRegistry[TransformerEvalConfig]()

transformer_eval_preset = transformer_eval_presets.decorator


@transformer_eval_preset("nllb_dense_600m")
def _nllb_dense_600m() -> TransformerEvalConfig:
    return TransformerEvalConfig()


def load_transformer_evaluator(
    config: TransformerEvalConfig, output_dir: Path
) -> Evaluator[Seq2SeqBatch]:
    """Load an :class:`Evaluator` for Transformer model evaluation."""
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
        log.info("Loading {} parallel text dataset.", dataset_card.name)

        dataset = load_parallel_text_dataset(dataset_card)

        log.info("Dataset loaded.")
    else:
        try:
            path = Path(config.dataset)
        except ValueError:
            raise AssetNotFoundError(
                config.dataset, f"An asset with the name '{config.dataset}' cannot be found."  # type: ignore[arg-type]
            )

        dataset = GenericParallelTextDataset.from_path(path)

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

    # Initialize the evaluation units.
    generator_max_num_batches = config.generator_max_num_batches

    if generator_max_num_batches is not None:
        if generator_max_num_batches % gang.size != 0:
            raise ValueError(
                f"`config.generator_max_num_batches` must be divisible by the size of the gang ({gang.size}), but is {generator_max_num_batches} instead."
            )

        generator_max_num_batches //= gang.size

    units: List[EvalUnit[Seq2SeqBatch]] = []

    data_readers = []

    for direction in dataset.directions(config.split):
        # Loss Evaluation
        loss_unit = TransformerLossEvalUnit(
            model,
            direction,
            gang,
            label_smoothing=config.label_smoothing,
        )

        units.append(loss_unit)

        data_reader = dataset.create_reader(
            config.split,
            tokenizer,
            gang,
            config.max_seq_len,
            batching=LengthBatching(config.max_num_tokens),
            direction=direction,
            num_prefetch=config.num_prefetch,
            seed=seed,
        )

        seed += 1

        data_readers.append(data_reader)

        # BLEU Evaluation
        text_output_file = output_dir.joinpath(
            f"translations/{direction}/rank_{gang.rank}.txt"
        )

        json_output_file = output_dir.joinpath(
            f"translations/{direction}/rank_{gang.rank}.jsonl"
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

        bleu_unit = TransformerBleuEvalUnit(
            direction,
            generator,
            tokenizer,
            gang,
            text_output_stream=text_output_fp,
            json_output_stream=json_output_fp,
        )

        units.append(bleu_unit)

        data_reader = dataset.create_reader(
            config.split,
            tokenizer,
            gang,
            config.max_seq_len,
            batching=StaticBatching(config.generator_batch_size),
            direction=direction,
            max_num_batches=generator_max_num_batches,
            num_prefetch=config.num_prefetch,
            seed=seed,
        )

        seed += 1

        data_readers.append(data_reader)

    # Initialize the evaluator.
    return Evaluator[Seq2SeqBatch](
        units=units,
        data_readers=data_readers,
        root_gang=gang,
        tb_dir=output_dir.joinpath("tb"),
        metrics_dir=output_dir.joinpath("metrics"),
        seed=seed,
        wall_watch=wall_watch,
    )


@final
class TransformerLossEvalUnit(AbstractEvalUnit[Seq2SeqBatch]):
    """Represents a Transformer model loss evaluation unit."""

    _label_smoothing: float
    _metric_bag: Seq2SeqMetricBag

    def __init__(
        self,
        model: Module,
        direction: Direction,
        gang: Gang,
        *,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(model, display_name=f"loss/{direction}")

        self._label_smoothing = label_smoothing

        self._metric_bag = Seq2SeqMetricBag(gang)

    @override
    def __call__(self, batch: Seq2SeqBatch) -> None:
        input_batch, target_batch = as_auto_regressive_input(batch)

        output = self._forward(input_batch)

        loss = output.compute_loss(
            target_batch.seqs, label_smoothing=self._label_smoothing
        )

        self._metric_bag.update_nll_loss(input_batch, loss.detach())

        self._metric_bag.update_batch_metrics(input_batch)

    def _forward(self, batch: Seq2SeqBatch) -> SequenceModelOutput:
        return self._model(batch)  # type: ignore[no-any-return]

    @property
    @override
    def metric_bag(self) -> Seq2SeqMetricBag:
        return self._metric_bag


@final
class TransformerBleuEvalUnit(AbstractEvalUnit[Seq2SeqBatch]):
    """Represents a Transformer model BLEU evaluation unit."""

    _converter: SequenceToTextConverter
    _text_output_stream: Optional[TextIO]
    _json_output_stream: Optional[TextIO]
    _metric_bag: Seq2SeqGenerationMetricBag

    def __init__(
        self,
        direction: Direction,
        generator: Seq2SeqGenerator,
        tokenizer: TextTokenizer,
        gang: Gang,
        *,
        text_output_stream: Optional[TextIO] = None,
        json_output_stream: Optional[TextIO] = None,
    ) -> None:
        super().__init__(generator.model, display_name=f"bleu/{direction}")

        self._converter = SequenceToTextConverter(
            generator, tokenizer, "translation", direction.target_lang
        )

        self._text_output_stream = text_output_stream
        self._json_output_stream = json_output_stream

        self._metric_bag = Seq2SeqGenerationMetricBag(gang)

        self._metric_bag.register_metric(
            "bleu", BleuMetric(device=gang.device), persistent=False
        )

    @override
    def __call__(self, batch: Seq2SeqBatch) -> None:
        if batch.example is None:
            raise ValueError("`batch.example` must not be `None`.")

        try:
            refs = batch.example["target_text"]
        except KeyError:
            raise ValueError("`batch.example` must contain a 'target_text' item.")

        hyps, output = self._converter.batch_convert(
            batch.source_seqs, batch.source_padding_mask
        )

        self._metric_bag.bleu.update(refs, hyps)

        self._metric_bag.update_batch_metrics(output, batch.num_source_elements())

        # Dump as text.
        if stream := self._text_output_stream:
            for ref, hyp in zip(refs, hyps):
                stream.write("REF: ")
                stream.write(ref)
                stream.write("\n")
                stream.write("HYP: ")
                stream.write(hyp)
                stream.write("\n\n")

            stream.flush()

        # Dump as JSON.
        if stream := self._json_output_stream:
            for ref, hyp in zip(refs, hyps):
                json.dump({"ref": ref, "hyp": hyp}, stream, indent=None)

                stream.write("\n")

            stream.flush()

    @property
    @override
    def metric_bag(self) -> Seq2SeqGenerationMetricBag:
        return self._metric_bag
