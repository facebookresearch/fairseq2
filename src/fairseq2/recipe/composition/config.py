# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TypeVar

from fairseq2.recipe.config import (
    CommonSection,
    EvaluatorSection,
    GangSection,
    GeneratorSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    ReferenceModelSection,
    RegimeSection,
    Seq2SeqGeneratorSection,
    SequenceGeneratorSection,
    TrainerSection,
)
from fairseq2.recipe.internal.config import _get_config_section
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver

SectionT = TypeVar("SectionT")


def register_config_section(
    container: DependencyContainer,
    name: str,
    kls: type[SectionT],
    *,
    keyed: bool = True,
) -> None:
    def get_section(resolver: DependencyResolver) -> SectionT:
        return _get_config_section(resolver, name, kls)

    container.register(kls, get_section, key=name if keyed else None, singleton=True)


def _register_config_sections(container: DependencyContainer) -> None:
    def register(kls: type[SectionT], name: str) -> None:
        register_config_section(container, name, kls, keyed=False)

    register(CommonSection, "common")
    register(EvaluatorSection, "evaluator")
    register(GangSection, "gang")
    register(GeneratorSection, "generator")
    register(LRSchedulerSection, "lr_scheduler")
    register(ModelSection, "model")
    register(ReferenceModelSection, "model")
    register(OptimizerSection, "optimizer")
    register(RegimeSection, "regime")
    register(Seq2SeqGeneratorSection, "seq2seq_generator")
    register(SequenceGeneratorSection, "seq_generator")
    register(TrainerSection, "trainer")
