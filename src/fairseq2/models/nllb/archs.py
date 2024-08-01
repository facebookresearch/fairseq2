# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.data import VocabularyInfo
from fairseq2.models.transformer import (
    TransformerConfig,
    transformer_arch,
    transformer_archs,
)
from fairseq2.nn.transformer import TransformerNormOrder


@transformer_arch("nllb_dense_300m")
def _dense_300m() -> TransformerConfig:
    config = _dense_1b()

    config.num_encoder_layers = 6
    config.num_decoder_layers = 6
    config.ffn_inner_dim = 1024 * 4
    config.dropout_p = 0.3

    return config


@transformer_arch("nllb_dense_600m")
def _dense_600m() -> TransformerConfig:
    config = _dense_1b()

    config.num_encoder_layers = 12
    config.num_decoder_layers = 12
    config.ffn_inner_dim = 1024 * 4

    return config


@transformer_arch("nllb_dense_1b")
def _dense_1b() -> TransformerConfig:
    config = transformer_archs.get("base")

    config.model_dim = 1024
    config.vocab_info = VocabularyInfo(
        size=256206, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=0
    )
    config.num_encoder_layers = 24
    config.num_decoder_layers = 24
    config.num_encoder_attn_heads = 16
    config.num_decoder_attn_heads = 16
    config.ffn_inner_dim = 1024 * 8
    config.norm_order = TransformerNormOrder.PRE

    return config


@transformer_arch("nllb_dense_3b")
def _dense_3b() -> TransformerConfig:
    config = _dense_1b()

    config.model_dim = 2048

    return config
