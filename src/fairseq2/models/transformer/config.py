# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch


@dataclass
class TransformerConfig:
    """Holds the configuration of a Transformer model.

    The default values correspond to the *base* Transformer model as described
    in Table 3 of :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.
    """

    max_seq_len: int = 1024
    """The expected maximum sequence length."""

    model_dim: int = 512
    """The dimensionality of the model (i.e. inputs and outputs)."""

    num_enc_layers: int = 6
    """The number of encoder layers."""

    num_dec_layers: int = 6
    """The number of decoder layers."""

    num_enc_attn_heads: int = 8
    """The number of attention heads in encoder layers."""

    num_dec_attn_heads: int = 8
    """The number of attention heads in decoder layers."""

    ffn_inner_dim: int = 2048
    """The dimensionality of inner layers in feed-forward networks."""

    dropout_p: float = 0.1
    """The dropout probability on outputs of embedding dictionaries, attention
    layers, and feed-forward networks."""

    legacy_pos_embed: bool = False
    """If ``True``, sinusoidal positional embeddings will be initialized in a
    way that is compatible with the original fairseq."""

    dtype: torch.dtype = torch.float32
    """The data type of model parameters and buffers."""
