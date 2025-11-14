# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.generation import Seq2SeqGenerator, SequenceGenerator
from fairseq2.recipe.internal.seq_generator import (
    _Seq2SeqGeneratorFactory,
    _SequenceGeneratorFactory,
)
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def _register_seq_generators(container: DependencyContainer) -> None:
    # SequenceGenerator
    def create_seq_generator(resolver: DependencyResolver) -> SequenceGenerator:
        generator_factory = resolver.resolve(_SequenceGeneratorFactory)

        return generator_factory.create()

    container.register(SequenceGenerator, create_seq_generator, singleton=True)

    container.register_type(_SequenceGeneratorFactory)

    # Seq2SeqGenerator
    def create_seq2seq_generator(resolver: DependencyResolver) -> Seq2SeqGenerator:
        generator_factory = resolver.resolve(_Seq2SeqGeneratorFactory)

        return generator_factory.create()

    container.register(Seq2SeqGenerator, create_seq2seq_generator, singleton=True)

    container.register_type(_Seq2SeqGeneratorFactory)
