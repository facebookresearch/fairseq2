# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from types import NoneType

from fairseq2.data.tokenizers import Tokenizer, TokenizerHubAccessor
from fairseq2.models import ModelHubAccessor
from fairseq2.models.gemma3n.config import GEMMA3N_FAMILY, Gemma3nConfig
from fairseq2.models.gemma3n.model import Gemma3nModel

get_gemma3n_model_hub = ModelHubAccessor(
    GEMMA3N_FAMILY, kls=Gemma3nModel, config_kls=Gemma3nConfig
)

get_gemma3n_tokenizer_hub = TokenizerHubAccessor(
    GEMMA3N_FAMILY, kls=Tokenizer, config_kls=NoneType
)

__all__ = [
    "GEMMA3N_FAMILY",
    "get_gemma3n_model_hub",
    "get_gemma3n_tokenizer_hub",
]
