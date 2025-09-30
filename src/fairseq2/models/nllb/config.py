# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.runtime.dependency import DependencyContainer

NLLB_FAMILY: Final = "nllb"


@dataclass(kw_only=True)
class NllbConfig:
    """Holds the configuration of an NLLB model."""

    model_dim: int = 1024
    """The dimensionality of the model."""

    max_seq_len: int = 1024
    """The maximum sequence length."""

    vocab_size: int = 256_206
    """The size of the vocabulary."""

    pad_idx: int = 0
    """The index of the PAD symbol in the vocabulary."""

    num_encoder_layers: int = 24
    """The number of encoder layers."""

    num_decoder_layers: int = 24
    """The number of decoder layers."""

    num_encoder_attn_heads: int = 16
    """The number of attention heads in encoder layers."""

    num_decoder_attn_heads: int = 16
    """The number of attention heads in decoder layers."""

    ffn_inner_dim: int = 1024 * 8
    """The dimensionality of inner projection layers in feed-forward networks."""

    dropout_p: float = 0.1
    """The dropout probability on outputs of Transformer layers."""


def register_nllb_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, NllbConfig)

    @arch("dense_300m")
    def dense_300m() -> NllbConfig:
        config = dense_1b()

        config.num_encoder_layers = 6
        config.num_decoder_layers = 6
        config.ffn_inner_dim = 1024 * 4
        config.dropout_p = 0.3

        return config

    @arch("dense_600m")
    def dense_600m() -> NllbConfig:
        config = dense_1b()

        config.num_encoder_layers = 12
        config.num_decoder_layers = 12
        config.ffn_inner_dim = 1024 * 4

        return config

    @arch("dense_1b")
    def dense_1b() -> NllbConfig:
        return NllbConfig()

    @arch("dense_3b")
    def dense_3b() -> NllbConfig:
        config = dense_1b()

        config.model_dim = 2048

        return config
