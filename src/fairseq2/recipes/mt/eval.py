# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TextIO, final

import torch
from torch.nn import Module

from fairseq2.assets import AssetNotFoundError, default_asset_store
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
from fairseq2.generation.encoder_decoder import BeamSearchConfig, generator_factories
from fairseq2.generation.text import SequenceToTextConverter
from fairseq2.logging import get_log_writer
from fairseq2.metrics.text import BleuMetric, ChrfMetric
from fairseq2.models import load_model
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.seq2seq import Seq2SeqBatch, as_auto_regressive_input
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.recipes.common_metrics import Seq2SeqGenerationMetricBag, Seq2SeqMetricBag
from fairseq2.recipes.evaluator import AbstractEvalUnit, Evaluator, EvalUnit
from fairseq2.recipes.utils.asset import (
    AssetReference,
    asset_as_path,
    retrieve_asset_card,
)
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import (
    broadcast_model,
    check_model_type,
    setup_root_gang,
)
from fairseq2.typing import META, DataClass, DataType, override
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass
class MTEvalConfig:
    """Holds the configuration of a machine translation evaluation task."""

    # Data
    dataset: AssetReference = "foo"  # TODO: change!
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
    model: AssetReference = "nllb-200_dense_distill_600m"
    """The name of the model to evaluate."""

    checkpoint_dir: Optional[Path] = None
    """The checkpoint directory containing models saved by :class:`FileCheckpointManager`."""

    dtype: DataType = torch.float16
    """The data type of the model."""

    # Loss
    label_smoothing: float = 0.1
    """The amount of label smoothing to apply while computing the loss."""

    # BLEU/chrF++
    generator: str = "beam_search"
    """The sequence generator."""

    generator_config: Optional[DataClass] = field(
        default_factory=lambda: BeamSearchConfig()
    )
    """The configuration of the sequence generator."""

    generator_batch_size: int = 8
    """The number of sentences per batch."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""


mt_eval_presets = ConfigRegistry[MTEvalConfig]()

mt_eval_preset = mt_eval_presets.decorator


@mt_eval_preset("nllb_dense_600m")
def _nllb_dense_600m() -> MTEvalConfig:
    return MTEvalConfig()


def load_mt_evaluator(
    config: MTEvalConfig, output_dir: Path
) -> Evaluator[Seq2SeqBatch]:
    """Load an :class:`Evaluator` for machine translation evaluation."""
    wall_watch = Stopwatch(start=True)

    if config.checkpoint_dir is not None:
        default_asset_store.metadata_providers.append(
            CheckpointModelMetadataProvider(config.checkpoint_dir)
        )

    gang = setup_root_gang(log)

    seed = config.seed

    model_card = retrieve_asset_card(config.model)

    # Load the tokenizer.
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
        dataset_path = asset_as_path(config.dataset)

        dataset = GenericParallelTextDataset.from_path(dataset_path)

    # Load the model.
    log.info("Loading {} model on rank 0.", model_card.name)

    if gang.rank == 0:
        init_device = gang.device
    else:
        init_device = META

    try:
        model = load_model(model_card, device=init_device, dtype=config.dtype)
    except ValueError as ex:
        raise ValueError(
            "The model cannot be initialized. See nested exception for details."
        ) from ex

    if not isinstance(model, EncoderDecoderModel):
        raise ValueError(
            f"The model must be of type `{EncoderDecoderModel}`, but is of type `{type(model)}` instead."
        )

    gang.barrier()

    log.info("Model loaded on rank 0.")

    # Distribute the model to all processes in the gang.
    if gang.size != 1:
        broadcast_model(model, gang, log)

    log_model(model, log)

    # Initialize the sequence generator.
    try:
        generator_factory = generator_factories.get(
            config.generator, config.generator_config
        )

        generator = generator_factory(model)
    except ValueError as ex:
        raise ValueError(
            "The sequence generator cannot be created. See nested exception for details."
        ) from ex

    # Initialize the evaluation units.
    units: List[EvalUnit[Seq2SeqBatch]] = []

    data_readers = []

    for direction in dataset.directions(config.split):
        # Loss Evaluation
        loss_unit = MTLossEvalUnit(
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
            sync_batches=False,
            num_prefetch=config.num_prefetch,
            seed=seed,
        )

        seed += 1

        data_readers.append(data_reader)

        # BLEU/chrF++ Evaluation
        src_output_file = output_dir.joinpath(
            f"translations/{direction}/rank_{gang.rank}.src.txt"
        )

        ref_output_file = output_dir.joinpath(
            f"translations/{direction}/rank_{gang.rank}.ref.txt"
        )

        hyp_output_file = output_dir.joinpath(
            f"translations/{direction}/rank_{gang.rank}.hyp.txt"
        )

        try:
            src_output_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise RuntimeError(
                f"The output directory '{src_output_file.parent}' cannot be created. See nested exception for details."
            ) from ex

        try:
            src_output_fp = src_output_file.open("w")
        except OSError as ex:
            raise RuntimeError(
                f"The output file '{src_output_file}' cannot be created. See nested exception for details."
            ) from ex

        try:
            ref_output_fp = ref_output_file.open("w")
        except OSError as ex:
            raise RuntimeError(
                f"The output file '{ref_output_file}' cannot be created. See nested exception for details."
            ) from ex

        try:
            hyp_output_fp = hyp_output_file.open("w")
        except OSError as ex:
            raise RuntimeError(
                f"The output file '{hyp_output_file}' cannot be created. See nested exception for details."
            ) from ex

        score_unit = MTBleuChrfEvalUnit(
            direction,
            generator,
            tokenizer,
            gang,
            src_output_stream=src_output_fp,
            ref_output_stream=ref_output_fp,
            hyp_output_stream=hyp_output_fp,
        )

        units.append(score_unit)

        data_reader = dataset.create_reader(
            config.split,
            tokenizer,
            gang,
            config.max_seq_len,
            batching=StaticBatching(config.generator_batch_size),
            direction=direction,
            sync_batches=False,
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
class MTLossEvalUnit(AbstractEvalUnit[Seq2SeqBatch]):
    """Represents a machine translation loss evaluation unit."""

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
        """
        :param model:
            The encoder-decoder model. Might be wrapped with DDP or FSDP.
        :param direction:
            The language direction to evaluate.
        :param gang:
            The gang for distributed evaluation.
        :param label_smoothing:
            The amount of label smoothing to apply while computing the loss.
        """
        super().__init__(model, display_name=f"loss/{direction}")

        check_model_type(model, EncoderDecoderModel)

        self._label_smoothing = label_smoothing

        self._metric_bag = Seq2SeqMetricBag(gang, train=False)

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
class MTBleuChrfEvalUnit(AbstractEvalUnit[Seq2SeqBatch]):
    """Represents a machine translation BLEU/chrF++ evaluation unit."""

    _converter: SequenceToTextConverter
    _src_output_stream: Optional[TextIO]
    _ref_output_stream: Optional[TextIO]
    _hyp_output_stream: Optional[TextIO]
    _metric_bag: Seq2SeqGenerationMetricBag

    def __init__(
        self,
        direction: Direction,
        generator: Seq2SeqGenerator,
        tokenizer: TextTokenizer,
        gang: Gang,
        *,
        src_output_stream: Optional[TextIO] = None,
        ref_output_stream: Optional[TextIO] = None,
        hyp_output_stream: Optional[TextIO] = None,
    ) -> None:
        """
        :param direction:
            The language direction to evaluate.
        :param generator:
            The sequence generator.
        :param tokenizer:
            The tokenizer to encode target text.
        :param gang:
            The gang for distributed evaluation.
        :param src_output_stream:
            The output stream to dump sentences in the source language.
        :param ref_output_stream:
            The output stream to dump references.
        :param hyp_output_stream:
            The output stream to dump hypotheses.
        """
        super().__init__(generator.model, display_name=f"score/{direction}")

        self._converter = SequenceToTextConverter(
            generator, tokenizer, "translation", direction.target_lang
        )

        self._src_output_stream = src_output_stream
        self._ref_output_stream = ref_output_stream
        self._hyp_output_stream = hyp_output_stream

        self._metric_bag = Seq2SeqGenerationMetricBag(gang)

        self._metric_bag.register_metric(
            "bleu", BleuMetric(device=gang.device), persistent=False
        )

        self._metric_bag.register_metric(
            "chrf", ChrfMetric(device=gang.device), persistent=False
        )

    @override
    def __call__(self, batch: Seq2SeqBatch) -> None:
        if batch.example is None:
            raise ValueError("`batch.example` must not be `None`.")

        try:
            srcs = batch.example["source_text"]
        except KeyError:
            raise ValueError(
                "`batch.example` must contain a 'source_text' item."
            ) from None

        try:
            refs = batch.example["target_text"]
        except KeyError:
            raise ValueError(
                "`batch.example` must contain a 'target_text' item."
            ) from None

        hyps, output = self._converter.batch_convert(
            batch.source_seqs, batch.source_padding_mask
        )

        self._metric_bag.bleu.update(refs, hyps)
        self._metric_bag.chrf.update(refs, hyps)

        self._metric_bag.update_batch_metrics(output, batch.num_source_elements())

        # Dump source sentences.
        if stream := self._src_output_stream:
            for src in srcs:
                stream.write(src)
                stream.write("\n")

            stream.flush()

        # Dump references.
        if stream := self._ref_output_stream:
            for ref in refs:
                stream.write(ref)
                stream.write("\n")

            stream.flush()

        # Dump hypotheses.
        if stream := self._hyp_output_stream:
            for hyp in hyps:
                stream.write(hyp)
                stream.write("\n")

            stream.flush()

    @property
    @override
    def metric_bag(self) -> Seq2SeqGenerationMetricBag:
        return self._metric_bag
