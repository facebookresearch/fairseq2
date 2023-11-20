# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.nn.transformer.attention import SDPA as SDPA
from fairseq2.nn.transformer.attention import NaiveSDPA as NaiveSDPA
from fairseq2.nn.transformer.attention import SDPAFactory as SDPAFactory
from fairseq2.nn.transformer.attention import TorchSDPA as TorchSDPA
from fairseq2.nn.transformer.attention import create_default_sdpa as create_default_sdpa
from fairseq2.nn.transformer.attention import (
    default_sdpa_factory as default_sdpa_factory,
)
from fairseq2.nn.transformer.attention import (
    set_default_sdpa_factory as set_default_sdpa_factory,
)
from fairseq2.nn.transformer.attention_mask import ALiBiMask as ALiBiMask
from fairseq2.nn.transformer.attention_mask import ALiBiMaskFactory as ALiBiMaskFactory
from fairseq2.nn.transformer.attention_mask import AttentionMask as AttentionMask
from fairseq2.nn.transformer.attention_mask import (
    AttentionMaskFactory as AttentionMaskFactory,
)
from fairseq2.nn.transformer.attention_mask import (
    CausalAttentionMask as CausalAttentionMask,
)
from fairseq2.nn.transformer.attention_mask import (
    CausalAttentionMaskFactory as CausalAttentionMaskFactory,
)
from fairseq2.nn.transformer.attention_mask import (
    CustomAttentionMask as CustomAttentionMask,
)
from fairseq2.nn.transformer.decoder import (
    DecoderLayerOutputHook as DecoderLayerOutputHook,
)
from fairseq2.nn.transformer.decoder import (
    StandardTransformerDecoder as StandardTransformerDecoder,
)
from fairseq2.nn.transformer.decoder import TransformerDecoder as TransformerDecoder
from fairseq2.nn.transformer.decoder_layer import (
    StandardTransformerDecoderLayer as StandardTransformerDecoderLayer,
)
from fairseq2.nn.transformer.decoder_layer import (
    TransformerDecoderLayer as TransformerDecoderLayer,
)
from fairseq2.nn.transformer.encoder import (
    EncoderLayerOutputHook as EncoderLayerOutputHook,
)
from fairseq2.nn.transformer.encoder import (
    StandardTransformerEncoder as StandardTransformerEncoder,
)
from fairseq2.nn.transformer.encoder import TransformerEncoder as TransformerEncoder
from fairseq2.nn.transformer.encoder_layer import (
    StandardTransformerEncoderLayer as StandardTransformerEncoderLayer,
)
from fairseq2.nn.transformer.encoder_layer import (
    TransformerEncoderLayer as TransformerEncoderLayer,
)
from fairseq2.nn.transformer.ffn import FeedForwardNetwork as FeedForwardNetwork
from fairseq2.nn.transformer.ffn import GLUFeedForwardNetwork as GLUFeedForwardNetwork
from fairseq2.nn.transformer.ffn import (
    StandardFeedForwardNetwork as StandardFeedForwardNetwork,
)
from fairseq2.nn.transformer.layer_norm import LayerNormFactory as LayerNormFactory
from fairseq2.nn.transformer.layer_norm import (
    create_standard_layer_norm as create_standard_layer_norm,
)
from fairseq2.nn.transformer.multihead_attention import AttentionState as AttentionState
from fairseq2.nn.transformer.multihead_attention import (
    AttentionStateFactory as AttentionStateFactory,
)
from fairseq2.nn.transformer.multihead_attention import (
    AttentionWeightHook as AttentionWeightHook,
)
from fairseq2.nn.transformer.multihead_attention import (
    AttentionWeightStoreHook as AttentionWeightStoreHook,
)
from fairseq2.nn.transformer.multihead_attention import (
    FullAttentionState as FullAttentionState,
)
from fairseq2.nn.transformer.multihead_attention import (
    LocalAttentionState as LocalAttentionState,
)
from fairseq2.nn.transformer.multihead_attention import (
    LocalAttentionStateFactory as LocalAttentionStateFactory,
)
from fairseq2.nn.transformer.multihead_attention import (
    MultiheadAttention as MultiheadAttention,
)
from fairseq2.nn.transformer.multihead_attention import (
    StandardMultiheadAttention as StandardMultiheadAttention,
)
from fairseq2.nn.transformer.multihead_attention import (
    StaticAttentionState as StaticAttentionState,
)
from fairseq2.nn.transformer.norm_order import (
    TransformerNormOrder as TransformerNormOrder,
)
from fairseq2.nn.transformer.relative_attention import (
    RelativePositionalEncoding as RelativePositionalEncoding,
)
from fairseq2.nn.transformer.relative_attention import (
    RelativePositionSDPA as RelativePositionSDPA,
)
from fairseq2.nn.transformer.shaw_attention import (
    ShawRelativePositionSDPA as ShawRelativePositionSDPA,
)
from fairseq2.nn.transformer.shaw_attention import (
    init_shaw_embedding as init_shaw_embedding,
)
