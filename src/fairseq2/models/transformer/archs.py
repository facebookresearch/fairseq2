# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.config_registry import ConfigRegistry
from fairseq2.models.transformer.factory import TransformerConfig

transformer_archs = ConfigRegistry[TransformerConfig]()

transformer_arch = transformer_archs.decorator


def _register_transformer_archs() -> None:
    @transformer_arch("base")
    def base() -> TransformerConfig:
        return TransformerConfig()

    @transformer_arch("big")
    def big() -> TransformerConfig:
        config = TransformerConfig()

        config.model_dim = 1024
        config.num_encoder_attn_heads = 16
        config.num_decoder_attn_heads = 16
        config.ffn_inner_dim = 4096
        config.dropout_p = 0.3

        return config
