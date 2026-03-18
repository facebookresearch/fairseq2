# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.data.tokenizers import Tokenizer, TokenizerHubAccessor
from fairseq2.models.hub import ModelHubAccessor
from fairseq2.models.olmo.config import OLMO_FAMILY, OLMOConfig
from fairseq2.models.olmo.tokenizer import OLMOTokenizerConfig as OLMOTokenizerConfig
from fairseq2.models.transformer_lm import TransformerLM

get_olmo_model_hub = ModelHubAccessor(
    OLMO_FAMILY, kls=TransformerLM, config_kls=OLMOConfig
)

get_olmo_tokenizer_hub = TokenizerHubAccessor(
    OLMO_FAMILY, kls=Tokenizer, config_kls=OLMOTokenizerConfig
)
