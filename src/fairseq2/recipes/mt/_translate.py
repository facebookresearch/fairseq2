# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO, final

import torch
from typing_extensions import override

from fairseq2.context import RuntimeContext
from fairseq2.data.text.tokenizers import TextTokenizer
from fairseq2.datasets import SequenceBatch, StaticBatching, SyncMode
from fairseq2.datasets.text import (
    GENERIC_TEXT_DATASET_FAMILY,
    TextDataset,
    TextReadOptions,
)
from fairseq2.device import CPU
from fairseq2.file_system import FileMode
from fairseq2.generation import BeamSearchConfig, Seq2SeqGenerator
from fairseq2.generation.text import SequenceToTextConverter
from fairseq2.metrics import MetricBag
from fairseq2.models.seq2seq import Seq2SeqModel
from fairseq2.recipes import (
    Generator,
    GeneratorUnit,
    Model,
    RecipeError,
    UnitError,
)
from fairseq2.recipes.common import (
    create_generator,
    create_seq2seq_generator,
    load_dataset,
    load_text_tokenizer,
    register_extra_asset_paths,
    setup_gangs,
    setup_reference_model,
    setup_torch,
)
from fairseq2.recipes.config import (
    CommonSection,
    DatasetSection,
    GangSection,
    GeneratorSection,
    ReferenceModelSection,
    Seq2SeqGeneratorSection,
    TextTokenizerSection,
)
from fairseq2.recipes.metrics import update_seq2seq_generator_metrics
from fairseq2.utils.rng import manual_seed
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@dataclass(kw_only=True)
class TextTranslateConfig:
    model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(
            name="nllb-200_dense_distill_600m"
        )
    )

    dataset: TextTranslateDatasetSection = field(
        default_factory=lambda: TextTranslateDatasetSection()
    )

    source_tokenizer: TextTokenizerSection = field(
        default_factory=lambda: TextTokenizerSection(name="nllb-200")
    )

    target_tokenizer: TextTokenizerSection = field(
        default_factory=lambda: TextTokenizerSection(name="nllb-200")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    source_lang: str = "eng_Latn"
    """The code of the language to translate from."""

    target_lang: str = "deu_Latn"
    """The code of the language to translate to."""

    generator: GeneratorSection = field(
        default_factory=lambda: GeneratorSection(dtype=torch.float16)
    )

    seq2seq_generator: Seq2SeqGeneratorSection = field(
        default_factory=lambda: Seq2SeqGeneratorSection(
            config=BeamSearchConfig(max_gen_len=(1, 256), echo_prompt=True),
            batch_size=8,
        )
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())


@dataclass(kw_only=True)
class TextTranslateDatasetSection(DatasetSection):
    name: str = "foo"  # TODO: change!

    family: str = GENERIC_TEXT_DATASET_FAMILY

    path: Path | None = None

    min_seq_len: int = 1
    """The minimum sequence length."""

    max_seq_len: int = 512
    """The maximum sequence length."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""


def register_text_translate_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(TextTranslateConfig)

    preset = registry.decorator

    @preset("nllb_dense")
    def nllb_dense() -> TextTranslateConfig:
        return TextTranslateConfig()


@torch.inference_mode()
def load_text_translator(
    context: RuntimeContext, config: object, output_dir: Path
) -> Generator:
    config = structure(config, TextTranslateConfig)

    validate(config)

    register_extra_asset_paths(context, config.common.assets)

    setup_torch(context, config.common.torch, output_dir)

    gangs = setup_gangs(context, config.gang)

    seed = config.common.seed

    manual_seed(seed, CPU, gangs.root.device)

    seed += 1

    model = setup_reference_model(
        Seq2SeqModel,
        context,
        config.model,
        gangs,
        config.generator.dtype,
        config.generator.amp,
    )

    dataset = load_dataset(TextDataset, context, config.dataset, gangs)

    source_tokenizer = load_text_tokenizer(context, config.source_tokenizer)

    if config.source_tokenizer == config.target_tokenizer:
        target_tokenizer = source_tokenizer
    else:
        target_tokenizer = load_text_tokenizer(context, config.target_tokenizer)

    # Initialize the unit.
    seq2seq_generator = create_seq2seq_generator(
        context, config.seq2seq_generator, model, target_tokenizer.vocab_info
    )

    if gangs.tp.rank == 0:
        file_system = context.file_system

        rank = gangs.dp.rank

        src_lang = config.source_lang
        tgt_lang = config.target_lang

        try:
            src_file = output_dir.joinpath(
                f"translations/{src_lang}-{tgt_lang}/rank_{rank}.src.txt"
            )
            hyp_file = output_dir.joinpath(
                f"translations/{src_lang}-{tgt_lang}/rank_{rank}.hyp.txt"
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
                hyp_fp = file_system.open_text(hyp_file, mode=FileMode.WRITE)
            except OSError as ex:
                raise UnitError(
                    f"The '{hyp_file}' output file cannot be created. See the nested exception for details."
                ) from ex
        except UnitError as ex:
            raise RecipeError(
                "The generator unit cannot be initialized. See the nested exception for details."
            ) from ex
    else:
        src_fp = None
        hyp_fp = None

    unit = TextTranslationUnit(
        model, seq2seq_generator, target_tokenizer, config.target_lang, src_fp, hyp_fp
    )

    text_encoder = source_tokenizer.create_encoder(
        task="translation", lang=config.source_lang, mode="source"
    )

    batching = StaticBatching(config.seq2seq_generator.batch_size)

    read_options = TextReadOptions(
        batching=batching,
        sync_mode=SyncMode.UNTIL_LAST,
        num_prefetch=config.dataset.num_prefetch,
        seed=seed,
        extras=config.dataset.extras,
    )

    data_reader = dataset.create_reader(
        text_encoder,
        target_tokenizer.vocab_info.pad_idx,
        gangs.dp,
        config.dataset.min_seq_len,
        config.dataset.max_seq_len,
        read_options,
    )

    seed += 1

    return create_generator(
        context,
        config.generator,
        config.common,
        output_dir,
        unit,
        data_reader,
        gangs,
        seed,
        hyper_params=config,
    )


@final
class TextTranslationUnit(GeneratorUnit[SequenceBatch]):
    _model: Model
    _converter: SequenceToTextConverter
    _src_output_stream: TextIO | None
    _hyp_output_stream: TextIO | None

    def __init__(
        self,
        model: Model,
        generator: Seq2SeqGenerator,
        tokenizer: TextTokenizer,
        target_lang: str,
        src_output_stream: TextIO | None,
        hyp_output_stream: TextIO | None,
    ) -> None:
        """
        :param generator:
            The sequence generator.
        :param tokenizer:
            The tokenizer to encode target text.
        :param target_lang:
            The code of the language to translate to.
        :param gang:
            The gang for distributed translation.
        :param src_output_stream:
            The output stream to dump sentences in the source language.
        :param hyp_output_stream:
            The output stream to dump hypotheses.
        """
        self._model = model

        self._converter = SequenceToTextConverter(
            generator, tokenizer, task="translation", target_lang=target_lang
        )

        self._src_output_stream = src_output_stream
        self._hyp_output_stream = hyp_output_stream

    @override
    def __call__(self, batch: SequenceBatch, metric_bag: MetricBag) -> None:
        if batch.example is None:
            raise ValueError("`batch.example` must not be `None`.")

        if not isinstance(batch.example, Mapping):
            raise TypeError(
                f"`batch.example` must be of type `{Mapping}`, but is of type `{type(batch.example)}` instead."
            )

        try:
            srcs = batch.example["text"]
        except KeyError:
            raise ValueError("`batch.example` must contain a 'text' item.") from None

        if not isinstance(srcs, Sequence):
            raise TypeError(
                f"`batch.example['text'] must be a sequence of strings, but is of type `{type(srcs)}` instead."
            )

        seqs, seqs_layout = batch.as_input()

        hyps, output = self._converter.batch_convert(seqs, seqs_layout)

        update_seq2seq_generator_metrics(metric_bag, output, batch.num_elements)

        try:
            # Dump source sentences.
            stream = self._src_output_stream
            if stream is not None:
                for src in srcs:
                    stream.write(src)
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
    def model(self) -> Model:
        return self._model
