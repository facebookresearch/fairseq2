# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.models.s2t_transformer.arch import S2TTransformer as S2TTransformer
from fairseq2.models.s2t_transformer.arch import (
    TransformerFbankFrontend as TransformerFbankFrontend,
)
from fairseq2.models.s2t_transformer.builder import (
    S2TTransformerConfig as S2TTransformerConfig,
)
from fairseq2.models.s2t_transformer.builder import (
    build_s2t_transformer as build_s2t_transformer,
)
from fairseq2.models.s2t_transformer.subsampler import (
    Conv1dFbankSubsampler as Conv1dFbankSubsampler,
)
from fairseq2.models.s2t_transformer.subsampler import (
    FbankSubsampler as FbankSubsampler,
)


def s2t_transformer_config_tiny() -> S2TTransformerConfig:
    """Get the configuration of the tiny speech-to-text Transformer model."""
    return S2TTransformerConfig(
        model_dim=256,
        num_enc_attn_heads=4,
        num_dec_attn_heads=4,
        num_enc_layers=6,
        num_dec_layers=3,
        ffn_inner_dim=256 * 4,
        dropout_p=0.3,
    )


def s2t_transformer_config_small() -> S2TTransformerConfig:
    """Get the configuration of the small speech-to-text Transformer model."""
    return S2TTransformerConfig(
        model_dim=256,
        num_enc_attn_heads=4,
        num_dec_attn_heads=4,
        ffn_inner_dim=256 * 8,
        dropout_p=0.1,
    )


def s2t_transformer_config_medium() -> S2TTransformerConfig:
    """Get the configuration of the medium speech-to-text Transformer model."""
    return S2TTransformerConfig()


def s2t_transformer_config_large() -> S2TTransformerConfig:
    """Get the configuration of the large speech-to-text Transformer model."""
    return S2TTransformerConfig(
        model_dim=1024,
        num_enc_attn_heads=16,
        num_dec_attn_heads=16,
        ffn_inner_dim=1024 * 4,
        dropout_p=0.2,
    )
