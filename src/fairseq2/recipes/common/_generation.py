# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.context import RuntimeContext
from fairseq2.data import VocabularyInfo
from fairseq2.generation import (
    Seq2SeqGenerator,
    Seq2SeqGeneratorHandler,
    SequenceGenerator,
    SequenceGeneratorHandler,
    UnknownSeq2SeqGeneratorError,
    UnknownSequenceGeneratorError,
)
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.recipes import Model
from fairseq2.recipes.config import (
    Seq2SeqGeneratorSection,
    SequenceGeneratorSection,
)
from fairseq2.registry import Provider
from fairseq2.utils.structured import StructureError


def create_seq_generator(
    context: RuntimeContext,
    seq_generator_section: SequenceGeneratorSection,
    model: Model,
    vocab_info: VocabularyInfo,
) -> SequenceGenerator:
    seq_generator_handlers = context.get_registry(SequenceGeneratorHandler)

    creator = _SequenceGeneratorCreator(seq_generator_handlers)

    return creator.create(seq_generator_section, model, vocab_info)


@final
class _SequenceGeneratorCreator:
    _seq_generator_handlers: Provider[SequenceGeneratorHandler]

    def __init__(
        self, seq_generator_handlers: Provider[SequenceGeneratorHandler]
    ) -> None:
        self._seq_generator_handlers = seq_generator_handlers

    def create(
        self,
        seq_generator_section: SequenceGeneratorSection,
        model: Model,
        vocab_info: VocabularyInfo,
    ) -> SequenceGenerator:
        if not isinstance(model.base_module, DecoderModel):
            raise TypeError(
                f"`model` must be of type `{DecoderModel}`, but is of type `{type(model)}` instead."
            )

        try:
            handler = self._seq_generator_handlers.get(seq_generator_section.name)
        except LookupError:
            raise UnknownSequenceGeneratorError(seq_generator_section.name) from None

        try:
            return handler.create(
                model.base_module, vocab_info, seq_generator_section.config
            )
        except StructureError as ex:
            raise StructureError(
                "`seq_generator.config` cannot be structured. See the nested exception for details."
            ) from ex


def create_seq2seq_generator(
    context: RuntimeContext,
    seq2seq_generator_section: Seq2SeqGeneratorSection,
    model: Model,
    target_vocab_info: VocabularyInfo,
) -> Seq2SeqGenerator:
    seq2seq_generator_handlers = context.get_registry(Seq2SeqGeneratorHandler)

    creator = _Seq2SeqGeneratorCreator(seq2seq_generator_handlers)

    return creator.create(seq2seq_generator_section, model, target_vocab_info)


@final
class _Seq2SeqGeneratorCreator:
    _seq2seq_generator_handlers: Provider[Seq2SeqGeneratorHandler]

    def __init__(
        self, seq2seq_generator_handlers: Provider[Seq2SeqGeneratorHandler]
    ) -> None:
        self._seq2seq_generator_handlers = seq2seq_generator_handlers

    def create(
        self,
        seq2seq_generator_section: Seq2SeqGeneratorSection,
        model: Model,
        target_vocab_info: VocabularyInfo,
    ) -> Seq2SeqGenerator:
        if not isinstance(model.base_module, EncoderDecoderModel):
            raise TypeError(
                f"`model` must be of type `{EncoderDecoderModel}`, but is of type `{type(model)}` instead."
            )

        try:
            handler = self._seq2seq_generator_handlers.get(
                seq2seq_generator_section.name
            )
        except LookupError:
            raise UnknownSeq2SeqGeneratorError(seq2seq_generator_section.name) from None

        try:
            return handler.create(
                model.base_module, target_vocab_info, seq2seq_generator_section.config
            )
        except StructureError as ex:
            raise StructureError(
                "`seq2seq_generator.config` cannot be structured. See the nested exception for details."
            ) from ex
