# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.data.tokenizers import TokenizerHubAccessor
from fairseq2.models import ModelHubAccessor
from fairseq2.models.s2t_transformer.config import (
    S2T_TRANSFORMER_FAMILY,
    S2TTransformerConfig,
)
from fairseq2.models.s2t_transformer.tokenizer import (
    S2TTransformerTokenizer,
    S2TTransformerTokenizerConfig,
)
from fairseq2.models.transformer import TransformerModel

get_s2t_transformer_model_hub = ModelHubAccessor(
    S2T_TRANSFORMER_FAMILY, kls=TransformerModel, config_kls=S2TTransformerConfig
)

get_s2t_transformer_tokenizer_hub = TokenizerHubAccessor(
    S2T_TRANSFORMER_FAMILY,
    kls=S2TTransformerTokenizer,
    config_kls=S2TTransformerTokenizerConfig,
)
