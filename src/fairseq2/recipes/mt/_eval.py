# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, TextIO, final

import torch
from typing_extensions import override

from fairseq2.context import RuntimeContext
from fairseq2.data.text.tokenizers import TextTokenizer
from fairseq2.datasets import Batching, LengthBatching, StaticBatching, SyncMode
from fairseq2.datasets.parallel_text import (
    GENERIC_PARALLEL_TEXT_DATASET_FAMILY,
    Direction,
    ParallelTextDataset,
    ParallelTextReadOptions,
)
from fairseq2.gang import Gangs
from fairseq2.generation import BeamSearchConfig, Seq2SeqGenerator
from fairseq2.generation.text import SequenceToTextConverter
from fairseq2.metrics.text import DEFAULT_BLEU_TOKENIZER, BleuMetric, ChrfMetric
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.recipes import (
    Evaluator,
    EvalUnit,
    Model,
    RecipeError,
    Seq2SeqGenerationMetricBag,
    Seq2SeqMetricBag,
    UnitError,
)
from fairseq2.recipes.common import (
    create_evaluator,
    create_seq2seq_generator,
    load_dataset,
    load_text_tokenizer,
    register_extra_asset_paths,
    setup_gangs,
    setup_reference_model,
)
from fairseq2.recipes.config import (
    CommonSection,
    DatasetSection,
    EvaluatorSection,
    GangSection,
    ReferenceModelSection,
    Seq2SeqGeneratorSection,
)
from fairseq2.recipes.mt._common import MTCriterion, MTLossSection
from fairseq2.typing import CPU
from fairseq2.utils.file import FileMode
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@dataclass(kw_only=True)
class MTEvalConfig:
    model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(
            name="nllb-200_dense_distill_600m"
        )
    )

    dataset: MTEvalDatasetSection = field(
        default_factory=lambda: MTEvalDatasetSection()
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    evaluator: EvaluatorSection = field(
        default_factory=lambda: EvaluatorSection(dtype=torch.float16)
    )

    loss: MTLossSection = field(default_factory=lambda: MTLossSection())

    bleu_tokenizer: str = DEFAULT_BLEU_TOKENIZER

    seq2seq_generator: Seq2SeqGeneratorSection = field(
        default_factory=lambda: Seq2SeqGeneratorSection(
            config=BeamSearchConfig(max_gen_len=(1, 256), echo_prompt=True),
            batch_size=8,
        )
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())


@dataclass(kw_only=True)
class MTEvalDatasetSection(DatasetSection):
    name: str = "foo"  # TODO: change!

    family: str = GENERIC_PARALLEL_TEXT_DATASET_FAMILY

    path: Path | None = None

    split: str = "test"

    min_seq_len: int = 1
    """The maximum sequence length."""

    max_seq_len: int = 512
    """The maximum sequence length."""

    max_num_tokens: int = 4096
    """The maximum number of tokens per batch."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""


def register_mt_eval_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(MTEvalConfig)

    preset = registry.decorator

    @preset("nllb_dense")
    def nllb_dense() -> MTEvalConfig:
        return MTEvalConfig()


@torch.inference_mode()
def load_mt_evaluator(
    context: RuntimeContext, config: object, output_dir: Path
) -> Evaluator[Seq2SeqBatch]:
    config = structure(config, MTEvalConfig)

    validate(config)

    register_extra_asset_paths(context, config)

    torch.set_float32_matmul_precision("high")

    gangs = setup_gangs(context, config)

    seed = config.common.seed

    manual_seed(seed, CPU, gangs.root.device)

    seed += 1

    model = setup_reference_model(
        EncoderDecoderModel,
        context,
        config.model.name,
        gangs,
        config.evaluator.dtype,
        config.evaluator.amp,
        config.evaluator.torch_compile,
    )

    dataset = load_dataset(ParallelTextDataset, context, config, gangs)

    tokenizer = load_text_tokenizer(context, config)

    # Initialize the units.
    seq2seq_generator = create_seq2seq_generator(context, config, model)

    criterion = MTCriterion(model, label_smoothing=config.loss.label_smoothing)

    units: list[EvalUnit[Seq2SeqBatch]] = []

    data_readers = []

    for direction in dataset.directions(config.dataset.split):
        loss_unit = MTLossEvalUnit(criterion, direction, gangs)

        units.append(loss_unit)

        batching: Batching = LengthBatching(config.dataset.max_num_tokens)

        read_options = ParallelTextReadOptions(
            batching=batching,
            direction=direction,
            sync_mode=SyncMode.UNTIL_LAST,
            num_prefetch=config.dataset.num_prefetch,
            seed=seed,
            extras=config.dataset.extras,
        )

        data_reader = dataset.create_reader(
            config.dataset.split,
            tokenizer,
            gangs.dp,
            config.dataset.min_seq_len,
            config.dataset.max_seq_len,
            read_options,
        )

        seed += 1

        data_readers.append(data_reader)

        # BLEU/chrF++ Evaluation
        if gangs.tp.rank == 0:
            file_system = context.file_system

            rank = gangs.dp.rank

            try:
                src_file = output_dir.joinpath(
                    f"translations/{direction}/rank_{rank}.src.txt"
                )
                ref_file = output_dir.joinpath(
                    f"translations/{direction}/rank_{rank}.ref.txt"
                )
                hyp_file = output_dir.joinpath(
                    f"translations/{direction}/rank_{rank}.hyp.txt"
                )

                try:
                    file_system.make_directory(src_file.parent)
                except OSError as ex:
                    raise UnitError(
                        f"The '{src_file.parent}' output directory cannot be created. See the nested exception for details."
                    ) from ex

                try:
                    src_fp = file_system.open_text(src_file, mode=FileMode.WRITE)
                except OSError as ex:
                    raise UnitError(
                        f"The '{src_file}' output file cannot be created. See the nested exception for details."
                    ) from ex

                try:
                    ref_fp = file_system.open_text(ref_file, mode=FileMode.WRITE)
                except OSError as ex:
                    raise UnitError(
                        f"The '{ref_file}' output file cannot be created. See the nested exception for details."
                    ) from ex

                try:
                    hyp_fp = file_system.open_text(hyp_file, mode=FileMode.WRITE)
                except OSError as ex:
                    raise UnitError(
                        f"The '{hyp_file}' output file cannot be created. See the nested exception for details."
                    ) from ex
            except UnitError as ex:
                raise RecipeError(
                    f"The 'score/{direction}' evaluator unit cannot be initialized. See the nested exception for details."
                ) from ex
        else:
            src_fp = None
            ref_fp = None
            hyp_fp = None

        score_unit = MTBleuChrfEvalUnit(
            model,
            direction,
            seq2seq_generator,
            tokenizer,
            gangs,
            src_output_stream=src_fp,
            ref_output_stream=ref_fp,
            hyp_output_stream=hyp_fp,
            bleu_tokenizer=config.bleu_tokenizer,
        )

        units.append(score_unit)

        batching = StaticBatching(config.seq2seq_generator.batch_size)

        read_options = ParallelTextReadOptions(
            direction=direction,
            batching=batching,
            sync_mode=SyncMode.UNTIL_LAST,
            num_prefetch=config.dataset.num_prefetch,
            seed=seed,
            extras=config.dataset.extras,
        )

        data_reader = dataset.create_reader(
            config.dataset.split,
            tokenizer,
            gangs.dp,
            config.dataset.min_seq_len,
            config.dataset.max_seq_len,
            read_options,
        )

        data_readers.append(data_reader)

        seed += 1

    return create_evaluator(
        context, config, output_dir, units, data_readers, gangs, seed
    )


@final
class MTLossEvalUnit(EvalUnit[Seq2SeqBatch]):
    _name: str
    _criterion: MTCriterion
    _metric_bag: Seq2SeqMetricBag

    def __init__(
        self, criterion: MTCriterion, direction: Direction, gangs: Gangs
    ) -> None:
        self._name = f"loss/{direction}"

        self._criterion = criterion

        self._metric_bag = Seq2SeqMetricBag(gangs.dp, train=False)

    @override
    def __call__(self, batch: Seq2SeqBatch) -> None:
        self._criterion(batch, self._metric_bag)

    @property
    @override
    def name(self) -> str | None:
        return self._name

    @property
    @override
    def model(self) -> Model:
        return self._criterion.model

    @property
    @override
    def metric_bag(self) -> Seq2SeqMetricBag:
        return self._metric_bag


@final
class MTBleuChrfEvalUnit(EvalUnit[Seq2SeqBatch]):
    """Represents a machine translation BLEU/chrF++ evaluation unit."""

    _name: str
    _model: Model
    _converter: SequenceToTextConverter
    _src_output_stream: TextIO | None
    _ref_output_stream: TextIO | None
    _hyp_output_stream: TextIO | None
    _metric_bag: Seq2SeqGenerationMetricBag

    def __init__(
        self,
        model: Model,
        direction: Direction,
        generator: Seq2SeqGenerator,
        tokenizer: TextTokenizer,
        gangs: Gangs,
        *,
        src_output_stream: TextIO | None = None,
        ref_output_stream: TextIO | None = None,
        hyp_output_stream: TextIO | None = None,
        bleu_tokenizer: str = DEFAULT_BLEU_TOKENIZER,
    ) -> None:
        """
        :param direction: The language direction to evaluate.
        :param generator: The sequence generator.
        :param tokenizer: The tokenizer to encode target text.
        :param gang: The gang for distributed evaluation.
        :param src_output_stream: The output stream to dump sentences in the
            source language.
        :param ref_output_stream: The output stream to dump references.
        :param hyp_output_stream: The output stream to dump hypotheses.
        """
        self._name = f"score/{direction}"

        self._model = model

        self._converter = SequenceToTextConverter(
            generator, tokenizer, "translation", direction.target_lang
        )

        self._src_output_stream = src_output_stream
        self._ref_output_stream = ref_output_stream
        self._hyp_output_stream = hyp_output_stream

        self._metric_bag = Seq2SeqGenerationMetricBag(gangs.dp)

        device = gangs.root.device

        bleu_metric = BleuMetric(device=device, tokenizer=bleu_tokenizer)
        chrf_metric = ChrfMetric(device=device)

        self._metric_bag.register_metric("bleu", bleu_metric, persistent=False)
        self._metric_bag.register_metric("chrf", chrf_metric, persistent=False)

    @override
    def __call__(self, batch: Seq2SeqBatch) -> None:
        if batch.example is None:
            raise ValueError("`batch.example` must not be `None`.")

        if not isinstance(batch.example, Mapping):
            raise TypeError(
                f"`batch.example` must be of type `{Mapping}`, but is of type `{type(batch.example)}` instead."
            )

        try:
            srcs = batch.example["source_text"]
        except KeyError:
            raise ValueError(
                "`batch.example` must contain a 'source_text' item."
            ) from None

        if not isinstance(srcs, Iterable):
            raise TypeError(
                f"`batch.example['source_text'] must be an iterable of strings, but is of type `{type(srcs)}` instead."
            )

        try:
            refs = batch.example["target_text"]
        except KeyError:
            raise ValueError(
                "`batch.example` must contain a 'target_text' item."
            ) from None

        if not isinstance(refs, Iterable):
            raise TypeError(
                f"`batch.example['target_text'] must be an iterable of strings, but is of type `{type(refs)}` instead."
            )

        hyps, output = self._converter.batch_convert(
            batch.source_seqs, batch.source_padding_mask
        )

        self._metric_bag.bleu.update(refs, hyps)
        self._metric_bag.chrf.update(refs, hyps)

        self._metric_bag.update_batch_metrics(output, batch.num_source_elements())

        try:
            # Dump source sentences.
            stream = self._src_output_stream
            if stream is not None:
                for src in srcs:
                    stream.write(src)
                    stream.write("\n")

                stream.flush()

            # Dump references.
            stream = self._ref_output_stream
            if stream is not None:
                for ref in refs:
                    stream.write(ref)
                    stream.write("\n")

                stream.flush()

            # Dump hypotheses.
            stream = self._hyp_output_stream
            if stream is not None:
                for hyp in hyps:
                    stream.write(hyp)
                    stream.write("\n")

                stream.flush()
        except OSError as ex:
            raise UnitError(
                "The generator output cannot be written. See the nested exception for details."
            ) from ex

    @property
    @override
    def name(self) -> str | None:
        return self._name

    @property
    @override
    def model(self) -> Model:
        return self._model

    @property
    @override
    def metric_bag(self) -> Seq2SeqGenerationMetricBag:
        return self._metric_bag
