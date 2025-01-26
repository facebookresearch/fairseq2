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
from fairseq2.datasets import StaticBatching, SyncMode
from fairseq2.datasets.text import (
    GENERIC_TEXT_DATASET_FAMILY,
    TextDataset,
    TextReadOptions,
)
from fairseq2.error import SetupError
from fairseq2.gang import Gangs
from fairseq2.generation import BeamSearchConfig, Seq2SeqGenerator
from fairseq2.generation.text import SequenceToTextConverter
from fairseq2.logging import log
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes.common import (
    broadcast_model,
    compile_eval_model,
    create_generator,
    create_seq2seq_generator,
    load_dataset,
    load_eval_model,
    load_text_tokenizer,
    register_extra_asset_paths,
    setup_gangs,
)
from fairseq2.recipes.config import (
    DatasetSection,
    GenerateRecipeConfig,
    GeneratorSection,
    Seq2SeqGeneratorSection,
)
from fairseq2.recipes.generator import AbstractGeneratorUnit, Generator
from fairseq2.recipes.metrics import Seq2SeqGenerationMetricBag
from fairseq2.recipes.utils.log import log_model
from fairseq2.typing import CPU
from fairseq2.utils.config import process_config
from fairseq2.utils.file import FileMode
from fairseq2.utils.rng import manual_seed


@dataclass(kw_only=True)
class TextTranslateConfig(GenerateRecipeConfig):
    """Holds the configuration of a text translation task."""

    model: str = "nllb-200_dense_distill_600m"

    dataset: TextTranslateDatasetSection = field(
        default_factory=lambda: TextTranslateDatasetSection()
    )

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


def register_text_translate_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(TextTranslateConfig)

    preset = registry.decorator

    @preset("nllb_dense_600m")
    def nllb_dense_600m() -> TextTranslateConfig:
        return TextTranslateConfig()


@torch.inference_mode()
def load_text_translator(
    context: RuntimeContext, config: TextTranslateConfig, output_dir: Path
) -> Generator[SequenceBatch]:
    register_extra_asset_paths(context, config.assets)

    process_config(context, config)

    gangs = setup_gangs(context, config.gang)

    dataset = load_dataset(TextDataset, context, config.dataset, gangs)

    tokenizer = load_text_tokenizer(context, config.model)

    seed = config.seed

    manual_seed(seed, CPU, context.device)

    seed += 1

    model = load_eval_model(
        EncoderDecoderModel,
        context,
        config.model,
        gangs,
        config.generator.dtype,
        mixed_precision=config.generator.amp,
    )

    broadcast_model(config.model, model, gangs)

    remove_parametrizations(model)

    log_model(log, model, gangs)

    if config.generator.torch_compile:
        model = compile_eval_model(context, config.model, model)

    # Initialize the unit.
    seq2seq_generator = create_seq2seq_generator(
        context, config.seq2seq_generator, model
    )

    if gangs.tp.rank == 0:
        file_system = context.file_system

        rank = gangs.dp.rank

        src_lang = config.source_lang
        tgt_lang = config.target_lang

        src_file = output_dir.joinpath(
            f"translations/{src_lang}-{tgt_lang}/rank_{rank}.src.txt"
        )
        hyp_file = output_dir.joinpath(
            f"translations/{src_lang}-{tgt_lang}/rank_{rank}.hyp.txt"
        )

        try:
            file_system.make_directory(src_file.parent)
        except OSError as ex:
            raise SetupError(
                f"The '{src_file.parent}' output directory cannot be created. See the nested exception for details."
            ) from ex

        try:
            src_fp = file_system.open_text(src_file, mode=FileMode.WRITE)
        except OSError as ex:
            raise SetupError(
                f"The '{src_file}' output file cannot be created. See the nested exception for details."
            ) from ex

        try:
            hyp_fp = file_system.open_text(hyp_file, mode=FileMode.WRITE)
        except OSError as ex:
            raise SetupError(
                f"The '{hyp_file}' output file cannot be created. See the nested exception for details."
            ) from ex
    else:
        src_fp = None
        hyp_fp = None

    unit = TextTranslationUnit(
        seq2seq_generator, tokenizer, config.target_lang, gangs, src_fp, hyp_fp
    )

    text_encoder = tokenizer.create_encoder(
        task="translation", lang=config.source_lang, mode="source"
    )

    batching = StaticBatching(config.seq2seq_generator.batch_size)

    read_options = TextReadOptions(
        batching=batching,
        sync_mode=SyncMode.UNTIL_LAST,
        num_prefetch=config.dataset.num_prefetch,
        seed=seed,
    )

    data_reader = dataset.create_reader(
        text_encoder,
        tokenizer.vocab_info.pad_idx,
        gangs.dp,
        config.dataset.min_seq_len,
        config.dataset.max_seq_len,
        read_options,
    )

    seed += 1

    return create_generator(context, config, output_dir, unit, data_reader, gangs, seed)


@final
class TextTranslationUnit(AbstractGeneratorUnit[SequenceBatch]):
    _converter: SequenceToTextConverter
    _src_output_stream: TextIO | None
    _hyp_output_stream: TextIO | None
    _metric_bag: Seq2SeqGenerationMetricBag

    def __init__(
        self,
        generator: Seq2SeqGenerator,
        tokenizer: TextTokenizer,
        target_lang: str,
        gangs: Gangs,
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
        super().__init__(generator.model)

        self._converter = SequenceToTextConverter(
            generator, tokenizer, task="translation", target_lang=target_lang
        )

        self._src_output_stream = src_output_stream
        self._hyp_output_stream = hyp_output_stream

        self._metric_bag = Seq2SeqGenerationMetricBag(gangs.dp)

    @override
    def __call__(self, batch: SequenceBatch) -> None:
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

        if not isinstance(srcs, Iterable):
            raise TypeError(
                f"`batch.example['text'] must be an iterable of strings, but is of type `{type(srcs)}` instead."
            )

        hyps, output = self._converter.batch_convert(batch.seqs, batch.padding_mask)

        self._metric_bag.update_batch_metrics(output, batch.num_elements())

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

    @property
    @override
    def metric_bag(self) -> Seq2SeqGenerationMetricBag:
        return self._metric_bag
