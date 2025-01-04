# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import get_runtime_context
from fairseq2.generation.generator import Seq2SeqGenerator, SequenceGenerator
from fairseq2.generation.handler import (
    Seq2SeqGeneratorHandler,
    Seq2SeqGeneratorNotFoundError,
    SequenceGeneratorHandler,
    SequenceGeneratorNotFoundError,
)
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.utils.config import process_config
from fairseq2.utils.structured import structure


def create_seq_generator(
    name: str, model: DecoderModel, config: object = None
) -> SequenceGenerator:
    context = get_runtime_context()

    registry = context.get_registry(SequenceGeneratorHandler)

    try:
        handler = registry.get(name)
    except LookupError:
        raise SequenceGeneratorNotFoundError(name) from None

    if config is None:
        try:
            config = handler.config_kls()
        except TypeError:
            raise ValueError(
                f"`config` must be specified for the '{name}' sequence generator."
            ) from None
    else:
        config = structure(config, handler.config_kls)

    process_config(config)

    return handler.create(model, config)


def create_seq2seq_generator(
    name: str, model: EncoderDecoderModel, config: object = None
) -> Seq2SeqGenerator:
    context = get_runtime_context()

    registry = context.get_registry(Seq2SeqGeneratorHandler)

    try:
        handler = registry.get(name)
    except LookupError:
        raise Seq2SeqGeneratorNotFoundError(name) from None

    if config is None:
        try:
            config = handler.config_kls()
        except TypeError:
            raise ValueError(
                f"`config` must be specified for the '{name}' sequence-to-sequence generator."
            ) from None
    else:
        config = structure(config, handler.config_kls)

    process_config(config)

    return handler.create(model, config)
