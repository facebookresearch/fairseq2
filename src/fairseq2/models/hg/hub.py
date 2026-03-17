# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hub integration for HuggingFace models and tokenizers."""

from __future__ import annotations

from transformers import PreTrainedModel

from fairseq2.data.tokenizers import TokenizerHubAccessor
from fairseq2.models.hg.config import HG_FAMILY, HuggingFaceModelConfig
from fairseq2.models.hg.tokenizer import HgTokenizer, HgTokenizerConfig
from fairseq2.models.hub import ModelHubAccessor

get_hg_model_hub = ModelHubAccessor(
    HG_FAMILY, kls=PreTrainedModel, config_kls=HuggingFaceModelConfig
)

get_hg_tokenizer_hub = TokenizerHubAccessor(
    HG_FAMILY, kls=HgTokenizer, config_kls=HgTokenizerConfig
)
