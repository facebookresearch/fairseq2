# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.transformer._attention import SDPA as SDPA
from fairseq2.models.transformer._attention import NaiveSDPA as NaiveSDPA
from fairseq2.models.transformer._attention import SDPAFactory as SDPAFactory
from fairseq2.models.transformer._attention import TorchSDPA as TorchSDPA
from fairseq2.models.transformer._attention import (
    create_default_sdpa as create_default_sdpa,
)
from fairseq2.models.transformer._attention import (
    default_sdpa_factory as default_sdpa_factory,
)
from fairseq2.models.transformer._attention import (
    enable_memory_efficient_torch_sdpa as enable_memory_efficient_torch_sdpa,
)
from fairseq2.models.transformer._attention import (
    set_default_sdpa_factory as set_default_sdpa_factory,
)
from fairseq2.models.transformer._attention_mask import (
    AbstractAttentionMask as AbstractAttentionMask,
)
from fairseq2.models.transformer._attention_mask import ALiBiMask as ALiBiMask
from fairseq2.models.transformer._attention_mask import (
    ALiBiMaskFactory as ALiBiMaskFactory,
)
from fairseq2.models.transformer._attention_mask import AttentionMask as AttentionMask
from fairseq2.models.transformer._attention_mask import (
    AttentionMaskFactory as AttentionMaskFactory,
)
from fairseq2.models.transformer._attention_mask import (
    CausalAttentionMask as CausalAttentionMask,
)
from fairseq2.models.transformer._attention_mask import (
    CausalAttentionMaskFactory as CausalAttentionMaskFactory,
)
from fairseq2.models.transformer._attention_mask import (
    CustomAttentionMask as CustomAttentionMask,
)
from fairseq2.models.transformer._checkpoint import (
    convert_transformer_checkpoint as convert_transformer_checkpoint,
)
from fairseq2.models.transformer._config import (
    TRANSFORMER_MODEL_FAMILY as TRANSFORMER_MODEL_FAMILY,
)
from fairseq2.models.transformer._config import TransformerConfig as TransformerConfig
from fairseq2.models.transformer._config import (
    register_transformer_configs as register_transformer_configs,
)
from fairseq2.models.transformer._decoder import (
    StandardTransformerDecoder as StandardTransformerDecoder,
)
from fairseq2.models.transformer._decoder import (
    TransformerDecoder as TransformerDecoder,
)
from fairseq2.models.transformer._decoder import (
    TransformerDecoderLayerHook as TransformerDecoderLayerHook,
)
from fairseq2.models.transformer._decoder_layer import (
    StandardTransformerDecoderLayer as StandardTransformerDecoderLayer,
)
from fairseq2.models.transformer._decoder_layer import (
    TransformerDecoderLayer as TransformerDecoderLayer,
)
from fairseq2.models.transformer._encoder import (
    StandardTransformerEncoder as StandardTransformerEncoder,
)
from fairseq2.models.transformer._encoder import (
    TransformerEncoder as TransformerEncoder,
)
from fairseq2.models.transformer._encoder import (
    TransformerEncoderLayerHook as TransformerEncoderLayerHook,
)
from fairseq2.models.transformer._encoder_layer import (
    StandardTransformerEncoderLayer as StandardTransformerEncoderLayer,
)
from fairseq2.models.transformer._encoder_layer import (
    TransformerEncoderLayer as TransformerEncoderLayer,
)
from fairseq2.models.transformer._factory import (
    TransformerFactory as TransformerFactory,
)
from fairseq2.models.transformer._factory import (
    create_transformer_model as create_transformer_model,
)
from fairseq2.models.transformer._factory import (
    init_transformer_final_projection as init_transformer_final_projection,
)
from fairseq2.models.transformer._ffn import (
    DauphinFeedForwardNetwork as DauphinFeedForwardNetwork,
)
from fairseq2.models.transformer._ffn import FeedForwardNetwork as FeedForwardNetwork
from fairseq2.models.transformer._ffn import (
    GLUFeedForwardNetwork as GLUFeedForwardNetwork,
)
from fairseq2.models.transformer._ffn import (
    StandardFeedForwardNetwork as StandardFeedForwardNetwork,
)
from fairseq2.models.transformer._frontend import (
    TransformerEmbeddingFrontend as TransformerEmbeddingFrontend,
)
from fairseq2.models.transformer._frontend import (
    TransformerFrontend as TransformerFrontend,
)
from fairseq2.models.transformer._model import TransformerModel as TransformerModel
from fairseq2.models.transformer._multihead_attention import (
    AttentionState as AttentionState,
)
from fairseq2.models.transformer._multihead_attention import (
    AttentionStateFactory as AttentionStateFactory,
)
from fairseq2.models.transformer._multihead_attention import (
    AttentionWeightHook as AttentionWeightHook,
)
from fairseq2.models.transformer._multihead_attention import (
    AttentionWeightStoreHook as AttentionWeightStoreHook,
)
from fairseq2.models.transformer._multihead_attention import (
    FullAttentionState as FullAttentionState,
)
from fairseq2.models.transformer._multihead_attention import (
    LocalAttentionState as LocalAttentionState,
)
from fairseq2.models.transformer._multihead_attention import (
    LocalAttentionStateFactory as LocalAttentionStateFactory,
)
from fairseq2.models.transformer._multihead_attention import (
    MultiheadAttention as MultiheadAttention,
)
from fairseq2.models.transformer._multihead_attention import (
    StandardMultiheadAttention as StandardMultiheadAttention,
)
from fairseq2.models.transformer._multihead_attention import (
    StaticAttentionState as StaticAttentionState,
)
from fairseq2.models.transformer._multihead_attention import (
    init_mha_output_projection as init_mha_output_projection,
)
from fairseq2.models.transformer._multihead_attention import (
    init_qkv_projection as init_qkv_projection,
)
from fairseq2.models.transformer._norm_order import (
    TransformerNormOrder as TransformerNormOrder,
)
from fairseq2.models.transformer._normalization import (
    LayerNormFactory as LayerNormFactory,
)
from fairseq2.models.transformer._normalization import (
    create_standard_layer_norm as create_standard_layer_norm,
)
from fairseq2.models.transformer._relative_attention import (
    RelativePositionalEncoding as RelativePositionalEncoding,
)
from fairseq2.models.transformer._relative_attention import (
    RelativePositionSDPA as RelativePositionSDPA,
)
from fairseq2.models.transformer._shaw_attention import (
    ShawRelativePositionSDPA as ShawRelativePositionSDPA,
)
from fairseq2.models.transformer._shaw_attention import (
    init_shaw_embedding as init_shaw_embedding,
)

# isort: split

from fairseq2.models import ModelHubAccessor

get_transformer_model_hub = ModelHubAccessor(TransformerModel, TransformerConfig)
