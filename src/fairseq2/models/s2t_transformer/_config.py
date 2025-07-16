# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from fairseq2.context import RuntimeContext

S2T_TRANSFORMER_MODEL_FAMILY: Final = "s2t_transformer"


@dataclass(kw_only=True)
class S2TTransformerConfig:
    """Holds the configuration of an S2T Transformer model.

    The default values correspond to the medium architecture as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`.
    """

    model_dim: int = 512
    """The dimensionality of the model."""

    max_source_seq_len: int = 1024
    """The maximum source sequence length after feature extraction."""

    num_fbank_channels: int = 80
    """The number of source log-mel filterbank channels."""

    max_target_seq_len: int = 1024
    """The maximum target sequence length."""

    target_vocab_size: int = 10_000
    """The size of the target vocabulary."""

    pad_idx: int = 1
    """The index of the PAD symbol in the target vocabulary."""

    use_relative_pos: bool = False
    """If ``True``, uses relative positional encodings for source sequences."""

    use_conformer: bool = False
    """If ``True``, uses Conformer blocks instead of encoder layers."""

    num_encoder_layers: int = 12
    """The number of encoder layers."""

    num_decoder_layers: int = 6
    """The number of decoder layers."""

    num_encoder_attn_heads: int = 8
    """The number of attention heads in encoder layers."""

    num_decoder_attn_heads: int = 8
    """The number of attention heads in decoder layers."""

    ffn_inner_dim: int = 512 * 4
    """The dimensionality of inner projection layers in feed-forward networks."""

    dropout_p: float = 0.15
    """The dropout probability on outputs of Transformer layers."""

    depthwise_conv_kernel_size: int = 0
    """The kernel size of depthwise convolutions in Conformer blocks."""


def register_s2t_transformer_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(S2TTransformerConfig)

    arch = registry.decorator

    @arch("tiny")
    def tiny() -> S2TTransformerConfig:
        config = medium()

        config.model_dim = 256
        config.num_encoder_layers = 6
        config.num_decoder_layers = 3
        config.num_encoder_attn_heads = 4
        config.num_decoder_attn_heads = 4
        config.ffn_inner_dim = 256 * 4
        config.dropout_p = 0.3

        return config

    @arch("small")
    def small() -> S2TTransformerConfig:
        config = medium()

        config.model_dim = 256
        config.num_encoder_attn_heads = 4
        config.num_decoder_attn_heads = 4
        config.ffn_inner_dim = 256 * 8
        config.dropout_p = 0.1

        return config

    @arch("medium")
    def medium() -> S2TTransformerConfig:
        return S2TTransformerConfig()

    @arch("large")
    def large() -> S2TTransformerConfig:
        config = medium()

        config.model_dim = 1024
        config.num_encoder_attn_heads = 16
        config.num_decoder_attn_heads = 16
        config.ffn_inner_dim = 1024 * 4
        config.dropout_p = 0.2

        return config

    @arch("conformer_medium")
    def conformer_medium() -> S2TTransformerConfig:
        return S2TTransformerConfig(
            model_dim=256,
            max_source_seq_len=6000,
            num_fbank_channels=80,
            max_target_seq_len=1024,
            target_vocab_size=181,
            use_relative_pos=False,
            use_conformer=True,
            num_encoder_layers=12,
            num_decoder_layers=6,
            num_encoder_attn_heads=4,
            num_decoder_attn_heads=8,
            ffn_inner_dim=512 * 4,
            dropout_p=0.1,
            depthwise_conv_kernel_size=31,
        )
