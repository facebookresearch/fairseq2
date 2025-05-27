# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.logging import log
from fairseq2.recipes import Model, RecipeError
from fairseq2.recipes.common._error import ModelCompilationNotSupportedError
from fairseq2.recipes.config import CompileOptionsSection
from fairseq2.typing import Compilable


def compile_model(model: Model, options_section: CompileOptionsSection) -> None:
    if not model.handler.supports_compilation:
        raise ModelCompilationNotSupportedError(model.name)

    if model.newly_initialized:
        log.info("Applying torch.compile() to the model.")
    else:
        log.info("Applying torch.compile() to '{}' model.", model.name)

    try:
        model.handler.compile(
            model.module,
            fullgraph=options_section.fullgraph,
            dynamic=options_section.dynamic,
            mode=options_section.mode,
            backend=options_section.backend,
            options=options_section.backend_options,
        )
    except RuntimeError as ex:
        raise RecipeError(
            "torch.compile() has failed. See the nested exception for details."
        ) from ex


def compile_object(obj: Compilable, options_section: CompileOptionsSection) -> None:
    try:
        obj.compile(
            fullgraph=options_section.fullgraph,
            dynamic=options_section.dynamic,
            mode=options_section.mode,
            backend=options_section.backend,
            options=options_section.backend_options,
        )
    except RuntimeError as ex:
        raise RecipeError(
            "torch.compile() has failed. See the nested exception for details."
        ) from ex
