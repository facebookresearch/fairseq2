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

MISTRAL_FAMILY: Final = "mistral"


@dataclass(kw_only=True)
class MistralConfig:
    """Holds the configuration of a Mistral model.

    The default values correspond to the base architecture as described in
    :cite:t:`https://doi.org/10.48550/arXiv.2310.06825`.
    """

    model_dim: int = 4096
    """The dimensionality of the model."""

    max_seq_len: int = 8192
    """The maximum sequence length."""

    vocab_size: int = 32_000
    """The size of the vocabulary."""

    pad_idx: int | None = None
    """The index of the PAD symbol in the vocabulary."""

    attn_window_len: int = 4096
    """The local attention window length."""

    num_layers: int = 32
    """The number of decoder layers."""

    num_attn_heads: int = 32
    """The number of attention heads in decoder layers."""

    num_key_value_heads: int = 8
    """The number of key/value heads for Grouped Query Attention."""

    ffn_inner_dim: int = 14336
    """The dimensionality of inner projection layers in feed-forward networks."""

    dropout_p: float = 0.0
    """The dropout probability on outputs of Transformer layers."""


def register_mistral_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, MistralConfig)

    @arch("7b")
    def _7b() -> MistralConfig:
        return MistralConfig()
