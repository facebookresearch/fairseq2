# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.data.tokenizers import TokenizerHubAccessor
from fairseq2.models import ModelHubAccessor
from fairseq2.models.qwen.config import QWEN_FAMILY, QwenConfig
from fairseq2.models.qwen.tokenizer import QwenTokenizerConfig
from fairseq2.models.transformer_lm import TransformerLM

get_qwen_model_hub = ModelHubAccessor(QWEN_FAMILY, TransformerLM, QwenConfig)

get_qwen_tokenizer_hub = TokenizerHubAccessor(QWEN_FAMILY, QwenTokenizerConfig)
