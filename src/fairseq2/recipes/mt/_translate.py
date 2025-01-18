# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO, final

import torch
from typing_extensions import override

from fairseq2.assets import AssetCardNotFoundError, default_asset_store
from fairseq2.checkpoint import FileCheckpointMetadataProvider
from fairseq2.context import RuntimeContext
from fairseq2.data.text.tokenizers import TextTokenizer, get_text_tokenizer_hub
from fairseq2.datasets import StaticBatching, SyncMode
from fairseq2.datasets.text import (
    GenericTextDataset,
    TextReadOptions,
    get_text_dataset_hub,
)
from fairseq2.gang import Gang
from fairseq2.generation import (
    BeamSearchConfig,
    Seq2SeqGenerator,
    create_seq2seq_generator,
)
from fairseq2.generation.text import SequenceToTextConverter
from fairseq2.logging import log
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.sequence import SequenceBatch
from fairseq2.recipes.generator import AbstractGeneratorUnit, Generator
from fairseq2.recipes.metrics import Seq2SeqGenerationMetricBag
from fairseq2.recipes.utils.asset import (
    AssetReference,
    asset_as_path,
    retrieve_asset_card,
)
from fairseq2.recipes.utils.log import log_model
from fairseq2.recipes.utils.setup import broadcast_model, load_model, setup_root_gang
from fairseq2.typing import META, DataType
from fairseq2.utils.profiler import Stopwatch


@dataclass(kw_only=True)
class MTTranslateConfig:
    """Holds the configuration of a text translation task."""

    # Data
    dataset: AssetReference = "foo"  # TODO: change!
    """The name, path, or path to the asset card of the text dataset."""

    source_lang: str = "eng_Latn"
    """The code of the language to translate from."""

    target_lang: str = "deu_Latn"
    """The code of the language to translate to."""

    min_seq_len: int = 1
    """The minimum sequence length."""

    max_seq_len: int = 512
    """The maximum sequence length."""

    batch_size: int = 1
    """The number of sentences per batch."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    # Model
    model: AssetReference = "nllb-200_dense_distill_600m"
    """The name of the model to translate with."""

    checkpoint_dir: Path | None = None
    """The checkpoint directory containing models saved by :class:`FileCheckpointManager`."""

    dtype: DataType = torch.float16
    """The data type of the model."""

    amp: bool = False
    """If ``True``, runs evaluation with ``torch.amp``."""

    # Generation
    generator: str = "beam_search"
    """The sequence generator."""

    generator_config: Any = field(
        default_factory=lambda: BeamSearchConfig(max_gen_len=(1, 256), echo_prompt=True)
    )
    """The configuration of the sequence generator."""

    # Misc
    seed: int = 2
    """The random number generator seed to use."""


def register_mt_translate_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(MTTranslateConfig)

    preset = registry.decorator

    @preset("nllb_dense_600m")
    def nllb_dense_600m() -> MTTranslateConfig:
        return MTTranslateConfig()


@torch.inference_mode()
def load_mt_translator(
    context: RuntimeContext, config: MTTranslateConfig, output_dir: Path
) -> Generator[SequenceBatch]:
    """Load a :class:`Generator` for text translation."""
    wall_watch = Stopwatch(start=True)

    if config.checkpoint_dir is not None:
        default_asset_store.metadata_providers.append(
            FileCheckpointMetadataProvider(config.checkpoint_dir)
        )

    gang = setup_root_gang(log)

    model_card = retrieve_asset_card(config.model)

    # Load the tokenizer.
    log.info("Loading {} tokenizer.", model_card.name)

    tokenizer_hub = get_text_tokenizer_hub()

    tokenizer = tokenizer_hub.load(model_card)

    log.info("Tokenizer loaded.")

    # Load the dataset.
    try:
        dataset_card = retrieve_asset_card(config.dataset)
    except AssetCardNotFoundError:
        dataset_card = None

    if dataset_card is not None:
        log.info("Loading {} text dataset.", dataset_card.name)

        dataset_hub = get_text_dataset_hub()

        dataset = dataset_hub.load(dataset_card)

        log.info("Dataset loaded.")
    else:
        dataset_path = asset_as_path(config.dataset)

        dataset = GenericTextDataset.from_path(dataset_path, "path")

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
        generator = create_seq2seq_generator(
            config.generator, model, config.generator_config
        )
    except ValueError as ex:
        raise ValueError(
            "The sequence generator cannot be created. See nested exception for details."
        ) from ex

    # Initialize the generator unit.
    src_output_file = output_dir.joinpath(
        f"translations/{config.source_lang}-{config.target_lang}/rank_{gang.rank}.src.txt"
    )

    hyp_output_file = output_dir.joinpath(
        f"translations/{config.source_lang}-{config.target_lang}/rank_{gang.rank}.hyp.txt"
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
        hyp_output_fp = hyp_output_file.open("w")
    except OSError as ex:
        raise RuntimeError(
            f"The output file '{hyp_output_file}' cannot be created. See nested exception for details."
        ) from ex

    unit = TextTranslationUnit(
        generator, tokenizer, config.target_lang, gang, src_output_fp, hyp_output_fp
    )

    text_encoder = tokenizer.create_encoder(
        task="translation", lang=config.source_lang, mode="source"
    )

    seed = config.seed

    batching = StaticBatching(config.batch_size)

    read_options = TextReadOptions(
        batching=batching,
        sync_mode=SyncMode.UNTIL_LAST,
        num_prefetch=config.num_prefetch,
        seed=seed,
    )

    try:
        data_reader = dataset.create_reader(
            text_encoder,
            tokenizer.vocab_info.pad_idx,
            gang,
            config.min_seq_len,
            config.max_seq_len,
            read_options,
        )
    except ValueError as ex:
        raise ValueError(
            "The data reader cannot be initialized. See nested exception for details."
        ) from ex

    seed += 1

    # Initialize the generator.
    return Generator[SequenceBatch](
        unit=unit,
        data_reader=data_reader,
        root_gang=gang,
        dtype=config.dtype,
        amp=config.amp,
        metrics_dir=output_dir.joinpath("metrics"),
        seed=seed,
        wall_watch=wall_watch,
    )


@final
class TextTranslationUnit(AbstractGeneratorUnit[SequenceBatch]):
    """Represents a text translation unit."""

    _converter: SequenceToTextConverter
    _src_output_stream: TextIO
    _hyp_output_stream: TextIO
    _metric_bag: Seq2SeqGenerationMetricBag

    def __init__(
        self,
        generator: Seq2SeqGenerator,
        tokenizer: TextTokenizer,
        target_lang: str,
        gang: Gang,
        src_output_stream: TextIO,
        hyp_output_stream: TextIO,
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

        self._metric_bag = Seq2SeqGenerationMetricBag(gang)

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

        for src in srcs:
            stream.write(src)
            stream.write("\n")

        stream.flush()

        # Dump hypotheses.
        stream = self._hyp_output_stream

        for hyp in hyps:
            stream.write(hyp)
            stream.write("\n")

        stream.flush()

    @property
    @override
    def metric_bag(self) -> Seq2SeqGenerationMetricBag:
        return self._metric_bag
