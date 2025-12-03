# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.logging import log
from fairseq2.recipe.config import CompileOptions
from fairseq2.recipe.error import ConfigError
from fairseq2.recipe.internal.model import _ModelHolder


def _compile_model(
    model_holder: _ModelHolder, section_name: str, options: CompileOptions
) -> None:
    if not model_holder.family.supports_compilation:
        raise ConfigError(
            f"Model specified in `{section_name}` section does not support torch.compile()."
        )

    log.info("Applying torch.compile() to the model.")

    try:
        model_holder.family.compile(
            model_holder.model,
            fullgraph=options.fullgraph,
            dynamic=options.dynamic,
            mode=options.mode,
            backend=options.backend,
            options=options.backend_options,
        )
    except RuntimeError as ex:
        raise OperationError(
            f"torch.compile() failed for the model specified in `{section_name}` section."
        ) from ex
