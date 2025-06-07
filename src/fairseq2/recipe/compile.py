# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.logging import log
from fairseq2.model.context import ModelContext
from fairseq2.recipe.config import CompileOptionsSection
from fairseq2.recipe.error import TorchCompileNotSupportedError


def _compile_model(
    model_context: ModelContext, options_section: CompileOptionsSection
) -> None:
    if not model_context.handler.supports_compilation:
        raise TorchCompileNotSupportedError(model_context.name)

    if model_context.newly_initialized:
        log.info("Applying torch.compile() to the model.")
    else:
        log.info("Applying torch.compile() to '{}' model.", model_context.name)

    try:
        model_context.handler.compile(
            model_context.model,
            fullgraph=options_section.fullgraph,
            dynamic=options_section.dynamic,
            mode=options_section.mode,
            backend=options_section.backend,
            options=options_section.backend_options,
        )
    except RuntimeError as ex:
        raise TorchCompileError(
            "torch.compile() has failed. See the nested exception for details."
        ) from ex


class TorchCompileError(Exception):
    pass
