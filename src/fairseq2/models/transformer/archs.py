# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.transformer.factory import TransformerConfig, transformer_arch


@transformer_arch("base")
def _base() -> TransformerConfig:
    return TransformerConfig()


@transformer_arch("big")
def _big() -> TransformerConfig:
    config = TransformerConfig()

    config.model_dim = 1024
    config.num_encoder_attn_heads = 16
    config.num_decoder_attn_heads = 16
    config.ffn_inner_dim = 4096
    config.dropout_p = 0.3

    return config
