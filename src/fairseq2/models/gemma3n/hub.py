# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.gemma3n.config import GEMMA3N_FAMILY


# TODO(Phase 2): Define model hub accessor once TransformerLM equivalent is implemented
# get_gemma3n_model_hub = ModelHubAccessor(
#     GEMMA3N_FAMILY, kls=Gemma3nModel, config_kls=Gemma3nConfig
# )

# TODO(Phase 2): Define tokenizer hub accessor
# get_gemma3n_tokenizer_hub = TokenizerHubAccessor(
#     GEMMA3N_FAMILY, kls=Tokenizer, config_kls=Gemma3nTokenizerConfig
# )

__all__ = [
    "GEMMA3N_FAMILY",
]
