# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from fairseq2.config_registry import ConfigRegistry

JEPA_FAMILY: Final = "jepa"


@dataclass(kw_only=True)
class JepaConfig:
    """
    Holds the configuration of a JEPA model.

    The default values correspond to the 'base' JEPA architecture.
    """

    encoder_config: JepaEncoderConfig = field(
        default_factory=lambda: JepaEncoderConfig()
    )
    """The configuration of the Vision Transformer encoder."""


@dataclass(kw_only=True)
class JepaEncoderConfig:
    model_dim: int = 768
    """The dimensionality of the model."""

    num_input_channels: int = 3
    """The number of input channels per frame."""

    input_dims: tuple[int, ...] = (224, 224)
    """
    The supported native dimensionality of inputs. Expected to be 2-dimensional
    (height, width) for images and 3-dimensional (depth, height, width) for
    videos.
    """

    patch_dims: tuple[int, ...] = (16, 16)
    """The dimensionality of patches to be extracted from inputs."""

    num_encoder_layers: int = 12
    """The number of encoder layers."""

    num_encoder_attn_heads: int = 12
    """The number of attention heads in encoder layers."""

    qkv_bias: bool = True
    """
    If ``True``, query, key, and value projections in multi-head attention
    layers will have an additive bias.
    """

    attn_dropout_p: float = 0.0
    """The dropout probability on attention weights."""

    ffn_inner_dim_ratio: float = 4.0
    """
    The ratio of the dimensionality of the inner projection layers in
    feed-forward networks to :attr:`model_dim`.
    """

    dropout_p: float = 0.0
    """The dropout probability on outputs of Transformer layers."""

    uniform_power: bool = False
    """
    If ``True``, each patch dimension will have equal representation in the
    produced positional encodings.
    """


jepa_archs = ConfigRegistry[JepaConfig]()

jepa_arch = jepa_archs.decorator
