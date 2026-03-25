# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.data.tokenizers import TokenizerHubAccessor
from fairseq2.models import ModelHubAccessor
from fairseq2.models.qwen.config import (
    QWEN_FAMILY,
    QWEN35_FAMILY,
    QWEN35_MOE_FAMILY,
    Qwen35Config,
    Qwen35MoeConfig,
    QwenConfig,
)
from fairseq2.models.qwen.tokenizer import QwenTokenizer, QwenTokenizerConfig
from fairseq2.models.transformer_lm import TransformerLM

get_qwen_model_hub = ModelHubAccessor(
    QWEN_FAMILY, kls=TransformerLM, config_kls=QwenConfig
)

get_qwen_tokenizer_hub = TokenizerHubAccessor(
    QWEN_FAMILY, kls=QwenTokenizer, config_kls=QwenTokenizerConfig
)

get_qwen35_model_hub = ModelHubAccessor(
    QWEN35_FAMILY, kls=TransformerLM, config_kls=Qwen35Config
)

get_qwen35_tokenizer_hub = TokenizerHubAccessor(
    QWEN35_FAMILY, kls=QwenTokenizer, config_kls=QwenTokenizerConfig
)

get_qwen35_moe_model_hub = ModelHubAccessor(
    QWEN35_MOE_FAMILY, kls=TransformerLM, config_kls=Qwen35MoeConfig
)

get_qwen35_moe_tokenizer_hub = TokenizerHubAccessor(
    QWEN35_MOE_FAMILY, kls=QwenTokenizer, config_kls=QwenTokenizerConfig
)
