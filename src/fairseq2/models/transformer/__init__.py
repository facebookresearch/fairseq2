# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.transformer._attention_bias import (
    ALiBiAttentionBias as ALiBiAttentionBias,
)
from fairseq2.models.transformer._attention_bias import AttentionBias as AttentionBias
from fairseq2.models.transformer._attention_bias import (
    AttentionBiasCache as AttentionBiasCache,
)
from fairseq2.models.transformer._attention_bias import (
    CausalAttentionBias as CausalAttentionBias,
)
from fairseq2.models.transformer._attention_bias import IdentityBias as IdentityBias
from fairseq2.models.transformer._attention_bias import (
    materialize_attention_bias as materialize_attention_bias,
)
from fairseq2.models.transformer._attention_bias import (
    maybe_get_attention_bias_tensor as maybe_get_attention_bias_tensor,
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
from fairseq2.models.transformer._hub import (
    get_transformer_model_hub as get_transformer_model_hub,
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
from fairseq2.models.transformer._sdpa._base import SDPA as SDPA
from fairseq2.models.transformer._sdpa._default import SDPAFactory as SDPAFactory
from fairseq2.models.transformer._sdpa._default import (
    create_default_sdpa as create_default_sdpa,
)
from fairseq2.models.transformer._sdpa._default import (
    get_default_sdpa_factory as get_default_sdpa_factory,
)
from fairseq2.models.transformer._sdpa._default import (
    set_default_sdpa_factory as set_default_sdpa_factory,
)
from fairseq2.models.transformer._sdpa._flash2 import Flash2SDPA as Flash2SDPA
from fairseq2.models.transformer._sdpa._flash3 import Flash3SDPA as Flash3SDPA
from fairseq2.models.transformer._sdpa._naive import NaiveSDPA as NaiveSDPA
from fairseq2.models.transformer._sdpa._naive import (
    naive_scaled_dot_product_attention as naive_scaled_dot_product_attention,
)
from fairseq2.models.transformer._sdpa._relative import (
    RelativePositionalEncoding as RelativePositionalEncoding,
)
from fairseq2.models.transformer._sdpa._relative import (
    RelativePositionSDPA as RelativePositionSDPA,
)
from fairseq2.models.transformer._sdpa._shaw import (
    ShawRelativePositionSDPA as ShawRelativePositionSDPA,
)
from fairseq2.models.transformer._sdpa._shaw import (
    init_shaw_embedding as init_shaw_embedding,
)
from fairseq2.models.transformer._sdpa._torch import TorchSDPA as TorchSDPA
