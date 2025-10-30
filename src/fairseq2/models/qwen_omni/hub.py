# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.data.tokenizers import TokenizerHubAccessor
from fairseq2.models import ModelHubAccessor
from fairseq2.models.qwen_omni.config import QWEN_OMNI_FAMILY, QwenOmniConfig
# from fairseq2.models.qwen_omni.processor import QwenOmniProcessor, QwenOmniProcessorConfig
from fairseq2.models.transformer_lm import TransformerLM

get_qwen_omni_model_hub = ModelHubAccessor(
    QWEN_OMNI_FAMILY, kls=TransformerLM, config_kls=QwenOmniConfig
)

# get_qwen_omni_processor_hub = TokenizerHubAccessor(
    # QWEN_OMNI_FAMILY, kls=QwenProcessor, config_kls=QwenOmniProcessorConfig
# )
