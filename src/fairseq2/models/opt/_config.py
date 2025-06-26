# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from fairseq2.context import RuntimeContext

OPT_MODEL_FAMILY: Final = "opt"


@dataclass(kw_only=True)
class OPTConfig:
    """Holds the configuration of a OPT model.

    The default values correspond to the base architecture as described in
    TODO
    """

    model_dim: int = 768
    """The dimensionality of the model."""

    max_seq_len: int = 2048 + 1
    """The maximum sequence length."""

    vocab_size: int = 50272
    """The size of the vocabulary."""

    pad_idx: int | None = 1
    """The index of the PAD symbol in the vocabulary."""

    attn_window_len: int = 2048
    """The local attention window length."""

    num_layers: int = 12
    """The number of decoder layers."""

    num_attn_heads: int = 12
    """The number of attention heads in decoder layers."""

    num_key_value_heads: int = 12
    """The number of key/value heads for Grouped Query Attention."""

    ffn_inner_dim: int = 3072
    """The dimensionality of inner projection layers in feed-forward networks."""

    dropout_p: float = 0.1
    """The dropout probability on outputs of Transformer layers."""


def register_opt_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(OPTConfig)

    arch = registry.decorator

    @arch("opt_125m")
    def opt_125m() -> OPTConfig:
        return OPTConfig()
