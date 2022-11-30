# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.nn.transformer.attention import (
    AttentionFunction,
    default_scaled_dot_product_attention,
)
from fairseq2.nn.transformer.attention_mask import (
    ALiBiAttentionMaskGenerator,
    AttentionMaskGenerator,
    CausalAttentionMaskGenerator,
)
from fairseq2.nn.transformer.builder import TransformerBuilder
from fairseq2.nn.transformer.decoder import (
    StandardTransformerDecoder,
    TransformerDecoder,
)
from fairseq2.nn.transformer.decoder_layer import (
    StandardTransformerDecoderLayer,
    TransformerDecoderLayer,
)
from fairseq2.nn.transformer.encoder import (
    StandardTransformerEncoder,
    TransformerEncoder,
)
from fairseq2.nn.transformer.encoder_layer import (
    StandardTransformerEncoderLayer,
    TransformerEncoderLayer,
)
from fairseq2.nn.transformer.ffn import FeedForwardNetwork, StandardFeedForwardNetwork
from fairseq2.nn.transformer.model import Transformer, UntiedScoreProjection
from fairseq2.nn.transformer.multihead_attention import (
    AttentionWeightHook,
    MultiheadAttention,
    MultiheadAttentionState,
    StandardMultiheadAttention,
    StoreAttentionWeights,
)
from fairseq2.nn.transformer.norm_order import TransformerNormOrder

__all__ = [
    "ALiBiAttentionMaskGenerator",
    "AttentionFunction",
    "AttentionMaskGenerator",
    "AttentionWeightHook",
    "CausalAttentionMaskGenerator",
    "FeedForwardNetwork",
    "MultiheadAttention",
    "MultiheadAttentionState",
    "StandardFeedForwardNetwork",
    "StandardMultiheadAttention",
    "StandardTransformerDecoder",
    "StandardTransformerDecoderLayer",
    "StandardTransformerEncoder",
    "StandardTransformerEncoderLayer",
    "StoreAttentionWeights",
    "Transformer",
    "TransformerBuilder",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "TransformerNormOrder",
    "UntiedScoreProjection",
    "default_scaled_dot_product_attention",
]
