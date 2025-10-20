# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.logging import log
from fairseq2.models import ModelFamily
from fairseq2.recipe.config import CompileOptions
from fairseq2.recipe.error import TorchCompileError, TorchCompileNotSupportedError
from fairseq2.recipe.model import RecipeModel


def _compile_model(
    model: RecipeModel, family: ModelFamily, options: CompileOptions
) -> None:
    if not family.supports_compilation:
        raise TorchCompileNotSupportedError()

    log.info("Applying torch.compile() to the model.")

    try:
        family.compile(
            model.module,
            fullgraph=options.fullgraph,
            dynamic=options.dynamic,
            mode=options.mode,
            backend=options.backend,
            options=options.backend_options,
        )
    except RuntimeError as ex:
        raise TorchCompileError() from ex
