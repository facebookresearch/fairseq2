# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from types import NoneType

from fairseq2.data.tokenizers import Tokenizer, TokenizerHubAccessor
from fairseq2.models import ModelHubAccessor
from fairseq2.models.mistral.config import MISTRAL_FAMILY, MistralConfig
from fairseq2.models.transformer_lm import TransformerLM

get_mistral_model_hub = ModelHubAccessor(
    MISTRAL_FAMILY, kls=TransformerLM, config_kls=MistralConfig
)

get_mistral_tokenizer_hub = TokenizerHubAccessor(
    MISTRAL_FAMILY, kls=Tokenizer, config_kls=NoneType
)
