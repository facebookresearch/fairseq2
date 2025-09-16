# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.data.tokenizers import Tokenizer, TokenizerHubAccessor
from fairseq2.models import ModelHubAccessor
from fairseq2.models.llama4.config import LLAMA4_FAMILY, Llama4Config
from fairseq2.models.llama4.tokenizer import Llama4TokenizerConfig
from fairseq2.models.transformer_lm import TransformerLM

get_llama4_model_hub = ModelHubAccessor(
    LLAMA4_FAMILY, kls=TransformerLM, config_kls=Llama4Config
)

get_llama4_tokenizer_hub = TokenizerHubAccessor(
    LLAMA4_FAMILY, kls=Tokenizer, config_kls=Llama4TokenizerConfig
)
