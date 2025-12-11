# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hub integration for HuggingFace models and tokenizers.

This module provides hub accessors that integrate HuggingFace models and
tokenizers with fairseq2's model hub system. This allows you to load
preconfigured HuggingFace models using the fairseq2 hub interface.

Hub Accessors:
    get_hg_model_hub: Access to HuggingFace model configurations
    get_hg_tokenizer_hub: Access to HuggingFace tokenizer configurations

Example:
    Load a model through the hub system::

        from fairseq2.models.hg.hub import get_hg_model_hub

        model_hub = get_hg_model_hub()
        model = model_hub.load_model("hg_qwen25_omni_3b")
"""

from __future__ import annotations

from transformers import PreTrainedModel

from fairseq2.data.tokenizers import TokenizerHubAccessor
from fairseq2.models.hub import ModelHubAccessor
from fairseq2.models.hg.config import HG_FAMILY, HuggingFaceModelConfig
from fairseq2.models.hg.tokenizer import HgTokenizer, HgTokenizerConfig

get_hg_model_hub = ModelHubAccessor(
    HG_FAMILY, kls=PreTrainedModel, config_kls=HuggingFaceModelConfig
)

get_hg_tokenizer_hub = TokenizerHubAccessor(
    HG_FAMILY, kls=HgTokenizer, config_kls=HgTokenizerConfig
)
