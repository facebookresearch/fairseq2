# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.nn.transformer._attention import SDPA as SDPA
from fairseq2.nn.transformer._attention import NaiveSDPA as NaiveSDPA
from fairseq2.nn.transformer._attention import SDPAFactory as SDPAFactory
from fairseq2.nn.transformer._attention import TorchSDPA as TorchSDPA
from fairseq2.nn.transformer._attention import (
    create_default_sdpa as create_default_sdpa,
)
from fairseq2.nn.transformer._attention import (
    default_sdpa_factory as default_sdpa_factory,
)
from fairseq2.nn.transformer._attention import (
    enable_memory_efficient_torch_sdpa as enable_memory_efficient_torch_sdpa,
)
from fairseq2.nn.transformer._attention import (
    set_default_sdpa_factory as set_default_sdpa_factory,
)
from fairseq2.nn.transformer._attention_mask import (
    AbstractAttentionMask as AbstractAttentionMask,
)
from fairseq2.nn.transformer._attention_mask import ALiBiMask as ALiBiMask
from fairseq2.nn.transformer._attention_mask import ALiBiMaskFactory as ALiBiMaskFactory
from fairseq2.nn.transformer._attention_mask import AttentionMask as AttentionMask
from fairseq2.nn.transformer._attention_mask import (
    AttentionMaskFactory as AttentionMaskFactory,
)
from fairseq2.nn.transformer._attention_mask import (
    CausalAttentionMask as CausalAttentionMask,
)
from fairseq2.nn.transformer._attention_mask import (
    CausalAttentionMaskFactory as CausalAttentionMaskFactory,
)
from fairseq2.nn.transformer._attention_mask import (
    CustomAttentionMask as CustomAttentionMask,
)
from fairseq2.nn.transformer._decoder import (
    DecoderLayerOutputHook as DecoderLayerOutputHook,
)
from fairseq2.nn.transformer._decoder import (
    StandardTransformerDecoder as StandardTransformerDecoder,
)
from fairseq2.nn.transformer._decoder import TransformerDecoder as TransformerDecoder
from fairseq2.nn.transformer._decoder_layer import (
    StandardTransformerDecoderLayer as StandardTransformerDecoderLayer,
)
from fairseq2.nn.transformer._decoder_layer import (
    TransformerDecoderLayer as TransformerDecoderLayer,
)
from fairseq2.nn.transformer._encoder import (
    EncoderLayerOutputHook as EncoderLayerOutputHook,
)
from fairseq2.nn.transformer._encoder import (
    StandardTransformerEncoder as StandardTransformerEncoder,
)
from fairseq2.nn.transformer._encoder import TransformerEncoder as TransformerEncoder
from fairseq2.nn.transformer._encoder_layer import (
    StandardTransformerEncoderLayer as StandardTransformerEncoderLayer,
)
from fairseq2.nn.transformer._encoder_layer import (
    TransformerEncoderLayer as TransformerEncoderLayer,
)
from fairseq2.nn.transformer._ffn import (
    DauphinFeedForwardNetwork as DauphinFeedForwardNetwork,
)
from fairseq2.nn.transformer._ffn import FeedForwardNetwork as FeedForwardNetwork
from fairseq2.nn.transformer._ffn import GLUFeedForwardNetwork as GLUFeedForwardNetwork
from fairseq2.nn.transformer._ffn import GLUFeedForwardNetworkV2 as GLUFeedForwardNetworkV2
from fairseq2.nn.transformer._ffn import (
    StandardFeedForwardNetwork as StandardFeedForwardNetwork,
)
from fairseq2.nn.transformer._layer_norm import LayerNormFactory as LayerNormFactory
from fairseq2.nn.transformer._layer_norm import (
    create_standard_layer_norm as create_standard_layer_norm,
)
from fairseq2.nn.transformer._multihead_attention import (
    AttentionState as AttentionState,
)
from fairseq2.nn.transformer._multihead_attention import (
    AttentionStateFactory as AttentionStateFactory,
)
from fairseq2.nn.transformer._multihead_attention import (
    AttentionWeightHook as AttentionWeightHook,
)
from fairseq2.nn.transformer._multihead_attention import (
    AttentionWeightStoreHook as AttentionWeightStoreHook,
)
from fairseq2.nn.transformer._multihead_attention import (
    FullAttentionState as FullAttentionState,
)
from fairseq2.nn.transformer._multihead_attention import (
    LocalAttentionState as LocalAttentionState,
)
from fairseq2.nn.transformer._multihead_attention import (
    LocalAttentionStateFactory as LocalAttentionStateFactory,
)
from fairseq2.nn.transformer._multihead_attention import (
    MultiheadAttention as MultiheadAttention,
)
from fairseq2.nn.transformer._multihead_attention import (
    StandardMultiheadAttention as StandardMultiheadAttention,
)
from fairseq2.nn.transformer._multihead_attention import (
    StaticAttentionState as StaticAttentionState,
)
from fairseq2.nn.transformer._multihead_attention import (
    init_mha_output_projection as init_mha_output_projection,
)
from fairseq2.nn.transformer._multihead_attention import (
    init_qkv_projection as init_qkv_projection,
)
from fairseq2.nn.transformer._norm_order import (
    TransformerNormOrder as TransformerNormOrder,
)
from fairseq2.nn.transformer._relative_attention import (
    RelativePositionalEncoding as RelativePositionalEncoding,
)
from fairseq2.nn.transformer._relative_attention import (
    RelativePositionSDPA as RelativePositionSDPA,
)
from fairseq2.nn.transformer._residual import (
    DropPathResidualConnect as DropPathResidualConnect,
)
from fairseq2.nn.transformer._residual import (
    NormFormerResidualConnect as NormFormerResidualConnect,
)
from fairseq2.nn.transformer._residual import ResidualConnect as ResidualConnect
from fairseq2.nn.transformer._residual import (
    ScaledResidualConnect as ScaledResidualConnect,
)
from fairseq2.nn.transformer._shaw_attention import (
    ShawRelativePositionSDPA as ShawRelativePositionSDPA,
)
from fairseq2.nn.transformer._shaw_attention import (
    init_shaw_embedding as init_shaw_embedding,
)
