# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from fairseq2.context import RuntimeContext
from fairseq2.data import VocabularyInfo
from fairseq2.nn.transformer import TransformerNormOrder

TRANSFORMER_MODEL_FAMILY: Final = "transformer"


@dataclass(kw_only=True)
class TransformerConfig:
    """Holds the configuration of a Transformer model.

    The default values correspond to the base architecture as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.
    """

    model_dim: int = 512
    """The dimensionality of the model."""

    max_seq_len: int = 1024
    """The maximum sequence length."""

    vocab_info: VocabularyInfo = field(
        default_factory=lambda: VocabularyInfo(
            size=32768, unk_idx=None, bos_idx=None, eos_idx=1, pad_idx=0
        )
    )
    """The vocabulary information."""

    num_encoder_layers: int = 6
    """The number of encoder layers."""

    num_decoder_layers: int = 6
    """The number of decoder layers."""

    num_encoder_attn_heads: int = 8
    """The number of attention heads in encoder layers."""

    num_decoder_attn_heads: int = 8
    """The number of attention heads in decoder layers."""

    ffn_inner_dim: int = 2048
    """The dimensionality of inner projection layers in feed-forward networks."""

    norm_order: TransformerNormOrder = TransformerNormOrder.POST
    """The Layer Normalization order."""

    dropout_p: float = 0.1
    """The dropout probability on outputs of Transformer layers."""


def register_transformer_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(TransformerConfig)

    arch = registry.decorator

    @arch("base")
    def base() -> TransformerConfig:
        return TransformerConfig()

    @arch("big")
    def big() -> TransformerConfig:
        config = base()

        config.model_dim = 1024
        config.num_encoder_attn_heads = 16
        config.num_decoder_attn_heads = 16
        config.ffn_inner_dim = 4096
        config.dropout_p = 0.3

        return config
