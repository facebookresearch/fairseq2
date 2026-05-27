# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hub accessors for Gemma 4 model and tokenizer."""

from __future__ import annotations

from types import NoneType

from fairseq2.data.tokenizers import Tokenizer, TokenizerHubAccessor
from fairseq2.models import ModelHubAccessor
from fairseq2.models.gemma4.config import GEMMA4_FAMILY, Gemma4Config
from fairseq2.models.gemma4.model import Gemma4Model

get_gemma4_model_hub = ModelHubAccessor(
    GEMMA4_FAMILY, kls=Gemma4Model, config_kls=Gemma4Config
)

get_gemma4_tokenizer_hub = TokenizerHubAccessor(
    GEMMA4_FAMILY, kls=Tokenizer, config_kls=NoneType
)

__all__ = [
    "GEMMA4_FAMILY",
    "get_gemma4_model_hub",
    "get_gemma4_tokenizer_hub",
]
