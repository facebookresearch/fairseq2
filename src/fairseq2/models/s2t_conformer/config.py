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

S2T_CONFORMER_FAMILY: Final = "s2t_conformer"


@dataclass(kw_only=True)
class S2TConformerConfig:
    """Holds the configuration of an S2T Conformer model."""

    model_dim: int = 256
    """The dimensionality of the model."""

    max_source_seq_len: int = 6000
    """The maximum source sequence length after feature extraction."""

    num_fbank_channels: int = 80
    """The number of source log-mel filterbank channels."""

    max_target_seq_len: int = 1024
    """The maximum target sequence length."""

    target_vocab_size: int = 181
    """The size of the target vocabulary."""

    pad_idx: int = 1
    """The index of the PAD symbol in the target vocabulary."""

    use_relative_pos: bool = False
    """If ``True``, uses relative positional encodings for source sequences."""

    num_encoder_layers: int = 12
    """The number of encoder layers."""

    num_decoder_layers: int = 6
    """The number of decoder layers."""

    num_encoder_attn_heads: int = 4
    """The number of attention heads in encoder layers."""

    num_decoder_attn_heads: int = 8
    """The number of attention heads in decoder layers."""

    depthwise_conv_kernel_size: int = 31
    """The kernel size of depthwise convolutions in Conformer blocks."""

    ffn_inner_dim: int = 512 * 4
    """The dimensionality of inner projection layers in feed-forward networks."""

    dropout_p: float = 0.1
    """The dropout probability on outputs of Transformer layers."""


def register_s2t_conformer_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, S2TConformerConfig)

    @arch("medium")
    def medium() -> S2TConformerConfig:
        return S2TConformerConfig()
