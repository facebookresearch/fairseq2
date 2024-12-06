# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.bestrq.factory import (
    BestRQConfig,
    BestRQEncoderConfig,
    bestrq_arch,
    bestrq_encoder_arch,
)
from fairseq2.nn.transformer import TransformerNormOrder


def register_archs() -> None:
    @bestrq_arch("base")
    def _base() -> BestRQConfig:
        return BestRQConfig()

    @bestrq_arch("large")
    def _large() -> BestRQConfig:
        config = _base()

        config.encoder_config.model_dim = 1024
        config.encoder_config.num_encoder_layers = 24
        config.encoder_config.num_encoder_attn_heads = 16
        config.encoder_config.ffn_inner_dim = 4096
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.layer_drop_p = 0.2
        config.quantized_dim = 768
        config.final_dim = 768

        return config

    @bestrq_encoder_arch("base")
    def _base_encoder() -> BestRQEncoderConfig:
        config = _base()

        return config.encoder_config

    @bestrq_encoder_arch("large")
    def _large_encoder() -> BestRQEncoderConfig:
        config = _large()

        return config.encoder_config