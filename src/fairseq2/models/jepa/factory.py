# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import GELU, Conv2d, Conv3d

from fairseq2.config_registry import ConfigRegistry
from fairseq2.models.jepa.model import JepaModel
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.models.vit import (
    Conv2dPatchFeatureExtractor,
    Conv3dPatchFeatureExtractor,
    PatchFeatureExtractor,
    StandardViTFrontend,
)
from fairseq2.nn import (
    InterpolatedPositionEncoder,
    LayerNorm,
    Linear,
    Sinusoidal2dPositionEncoder,
    Sinusoidal3dPositionEncoder,
    StandardLayerNorm,
)
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.nn.transformer.residual import DropPathResidualConnect
from fairseq2.typing import DataType, Device

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

    init_std: float = 0.02
    """
    The standard deviation to initialize weights and biases of projection and
    normalization layers.
    """

    dropout_p: float = 0.0
    """The dropout probability on outputs of Transformer layers."""

    droppath_p: float = 0.0
    """
    The probability of dropping sequences from outputs of multi-head attention
    and feed-forward network layers before adding residuals.
    """

    uniform_power: bool = False
    """
    If ``True``, each patch dimension will have equal representation in the
    produced positional encodings.
    """


jepa_archs = ConfigRegistry[JepaConfig]()

jepa_arch = jepa_archs.decorator


# TODO(balioglu): work in progress. Supports only vision encoder.
class JepaBuilder:
    """Builds modules of a JEPA model."""

    _config: JepaConfig
    _encoder_builder: JepaEncoderBuilder
    _device: Device | None
    _dtype: DataType | None

    def __init__(
        self,
        config: JepaConfig,
        encoder_builder: JepaEncoderBuilder | None = None,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        self._config = config

        if encoder_builder is None:
            encoder_builder = JepaEncoderBuilder(
                config.encoder_config, device=device, dtype=dtype
            )

        self._encoder_builder = encoder_builder

        self._device, self._dtype = device, dtype

    def build_model(self) -> JepaModel:
        encoder_frontend = self._encoder_builder.build_frontend()

        encoder = self._encoder_builder.build_encoder()

        return JepaModel(encoder_frontend, encoder)


class JepaEncoderBuilder:
    """Builds modules of a JEPA Vision Transformer encoder."""

    _config: JepaEncoderConfig
    _device: Device | None
    _dtype: DataType | None

    def __init__(
        self,
        config: JepaEncoderConfig,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        self._config = config

        self._device, self._dtype = device, dtype

    def build_frontend(self) -> TransformerFrontend:
        config = self._config

        if len(config.input_dims) != len(config.patch_dims):
            raise ValueError(
                f"The lengths of `input_dims` and `patch_dims` must match, but they are {len(config.input_dims)} and {len(config.patch_dims)} instead."
            )

        feature_extractor = self.build_feature_extractor()

        pos_encoder = self.build_position_encoder()

        return StandardViTFrontend(feature_extractor, pos_encoder)

    def build_feature_extractor(self) -> PatchFeatureExtractor:
        config = self._config

        init_std = config.init_std

        num_patch_dims = len(config.patch_dims)

        if num_patch_dims == 3:
            patch_3d_dims = cast(tuple[int, int, int], config.patch_dims)

            def init_conv3d(conv: Conv3d) -> None:
                init_truncated_normal(conv.weight, conv.bias, std=init_std)

            return Conv3dPatchFeatureExtractor(
                config.num_input_channels,
                config.model_dim,
                patch_3d_dims,
                init_fn=init_conv3d,
                device=self._device,
                dtype=self._dtype,
            )
        elif num_patch_dims == 2:
            patch_2d_dims = cast(tuple[int, int], config.patch_dims)

            def init_conv2d(conv: Conv2d) -> None:
                init_truncated_normal(conv.weight, conv.bias, std=init_std)

            return Conv2dPatchFeatureExtractor(
                config.num_input_channels,
                config.model_dim,
                patch_2d_dims,
                init_fn=init_conv2d,
                device=self._device,
                dtype=self._dtype,
            )
        else:
            raise ValueError(
                f"The length of `patch_dims` must be 2 or 3, but is {num_patch_dims} instead."
            )

    def build_position_encoder(self) -> InterpolatedPositionEncoder:
        config = self._config

        num_input_dims = len(config.input_dims)

        if num_input_dims == 3:
            input_3d_dims = cast(tuple[int, int, int], config.input_dims)
            patch_3d_dims = cast(tuple[int, int, int], config.patch_dims)

            d_input_dim, h_input_dim, w_input_dim = input_3d_dims
            d_patch_dim, h_patch_dim, w_patch_dim = patch_3d_dims

            grid_3d_dims = (
                (d_input_dim // d_patch_dim),
                (h_input_dim // h_patch_dim),
                (w_input_dim // w_patch_dim),
            )

            return Sinusoidal3dPositionEncoder(
                config.model_dim,
                grid_3d_dims,
                uniform_power=config.uniform_power,
                device=self._device,
            )
        elif num_input_dims == 2:
            input_2d_dims = cast(tuple[int, int], config.input_dims)
            patch_2d_dims = cast(tuple[int, int], config.patch_dims)

            h_input_dim, w_input_dim = input_2d_dims
            h_patch_dim, w_patch_dim = patch_2d_dims

            grid_2d_dims = (h_input_dim // h_patch_dim), (w_input_dim // w_patch_dim)

            return Sinusoidal2dPositionEncoder(
                config.model_dim, grid_2d_dims, device=self._device
            )
        else:
            raise ValueError(
                f"The length of `input_dims` must be 2 or 3, but is {num_input_dims} instead."
            )

    def build_encoder(self, num_layers: int | None = None) -> TransformerEncoder:
        config = self._config

        if num_layers is None:
            num_layers = config.num_encoder_layers

        layers = [self.build_encoder_layer(i) for i in range(num_layers)]

        return StandardTransformerEncoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.build_layer_norm,
            device=self._device,
            dtype=self._dtype,
        )

    def build_encoder_layer(self, layer_idx: int) -> TransformerEncoderLayer:
        config = self._config

        self_attn = self.build_attention(layer_idx)

        ffn = self.build_ffn(layer_idx)

        drop_path = DropPathResidualConnect(drop_p=config.droppath_p)

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.build_layer_norm,
            self_attn_residual=drop_path,
            ffn_residual=drop_path,
            device=self._device,
            dtype=self._dtype,
        )

    def build_attention(self, layer_idx: int) -> MultiheadAttention:
        config = self._config

        sdpa = create_default_sdpa(attn_dropout_p=config.attn_dropout_p)

        output_proj = self.build_mha_output_projection(layer_idx)

        return StandardMultiheadAttention(
            config.model_dim,
            config.num_encoder_attn_heads,
            sdpa=sdpa,
            bias=config.qkv_bias,
            output_proj=output_proj,
            device=self._device,
            dtype=self._dtype,
        )

    def build_mha_output_projection(self, layer_idx: int) -> Linear:
        config = self._config

        init_std = config.init_std

        def init_projection(proj: Linear) -> None:
            init_truncated_normal(proj.weight, proj.bias, std=init_std)

            with torch.no_grad():
                proj.weight.div_(math.sqrt(2.0 * (layer_idx + 1)))

        return Linear(
            config.model_dim,
            config.model_dim,
            bias=True,
            init_fn=init_projection,
            device=self._device,
            dtype=self._dtype,
        )

    def build_ffn(self, layer_idx: int) -> FeedForwardNetwork:
        config = self._config

        init_std = config.init_std

        def init_projection(proj: Linear) -> None:
            init_truncated_normal(proj.weight, proj.bias, std=init_std)

            with torch.no_grad():
                proj.weight.div_(math.sqrt(2.0 * (layer_idx + 1)))

        inner_dim = int(config.model_dim * config.ffn_inner_dim_ratio)

        return StandardFeedForwardNetwork(
            config.model_dim,
            inner_dim,
            bias=True,
            inner_activation=GELU(),
            proj_init_fn=init_projection,
            norm_order=TransformerNormOrder.PRE,
            device=self._device,
            dtype=self._dtype,
        )

    def build_layer_norm(
        self,
        model_dim: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> LayerNorm:
        config = self._config

        init_std = config.init_std

        def init_layer_norm(m: LayerNorm) -> None:
            if m.weight is not None:
                init_truncated_normal(m.weight, m.bias, std=init_std)

        return StandardLayerNorm(
            model_dim,
            bias=True,
            eps=1e-6,
            init_fn=init_layer_norm,
            device=device,
            dtype=dtype,
        )


def init_truncated_normal(
    weight: Tensor, bias: Tensor | None, *, std: float = 1.0
) -> None:
    nn.init.trunc_normal_(weight, std=std)

    if bias is not None:
        nn.init.zeros_(bias)


def create_jepa_model(
    config: JepaConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> JepaModel:
    return JepaBuilder(config, device=device, dtype=dtype).build_model()
