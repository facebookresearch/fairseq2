# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.data.tokenizers import Tokenizer, TokenizerHubAccessor
from fairseq2.models import ModelHubAccessor
from fairseq2.models.llama.config import LLAMA_FAMILY, LLaMAConfig
from fairseq2.models.llama.tokenizer import LLaMATokenizerConfig as LLaMATokenizerConfig
from fairseq2.models.transformer_lm import TransformerLM

get_llama_model_hub = ModelHubAccessor(
    LLAMA_FAMILY, kls=TransformerLM, config_kls=LLaMAConfig
)

get_llama_tokenizer_hub = TokenizerHubAccessor(
    LLAMA_FAMILY, kls=Tokenizer, config_kls=LLaMATokenizerConfig
)
