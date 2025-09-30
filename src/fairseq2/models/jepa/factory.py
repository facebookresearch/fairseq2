# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import GELU, Conv2d, Conv3d

from fairseq2.models.jepa.config import JepaConfig, JepaEncoderConfig
from fairseq2.models.jepa.model import JepaModel
from fairseq2.models.transformer import (
    FeedForwardNetwork,
    IdentityBias,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerFrontend,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.models.vit import (
    Conv2dPatchFeatureExtractor,
    Conv3dPatchFeatureExtractor,
    PatchFeatureExtractor,
    StandardViTFrontend,
)
from fairseq2.nn import (
    DropPathResidualConnect,
    InterpolatedPositionEncoder,
    LayerNorm,
    Linear,
    Sinusoidal2dPositionEncoder,
    Sinusoidal3dPositionEncoder,
    StandardLayerNorm,
)


def create_jepa_model(config: JepaConfig) -> JepaModel:
    return JepaFactory(config).create_model()


# TODO(balioglu): work in progress. Supports only vision encoder.
class JepaFactory:
    def __init__(self, config: JepaConfig) -> None:
        self._config = config

    def create_model(self) -> JepaModel:
        config = self._config

        encoder_frontend = self.create_encoder_frontend()

        encoder = self.create_encoder()

        return JepaModel(config.encoder_config.model_dim, encoder_frontend, encoder)

    def create_encoder_frontend(self) -> TransformerFrontend:
        config = self._config

        factory = JepaEncoderFactory(config.encoder_config)

        return factory.create_encoder_frontend()

    def create_encoder(self) -> TransformerEncoder:
        config = self._config

        factory = JepaEncoderFactory(config.encoder_config)

        return factory.create_encoder()


class JepaEncoderFactory:
    def __init__(self, config: JepaEncoderConfig) -> None:
        self._config = config

    def create_encoder_frontend(self) -> TransformerFrontend:
        patch_feature_extractor = self.create_patch_feature_extractor()

        pos_encoder = self.create_position_encoder()

        return StandardViTFrontend(patch_feature_extractor, pos_encoder)

    def create_patch_feature_extractor(self) -> PatchFeatureExtractor:
        config = self._config

        init_std = config.init_std

        num_patch_dims = len(config.patch_dims)

        if num_patch_dims == 3:
            patch_3d_dims = cast(tuple[int, int, int], config.patch_dims)

            def init_conv3d(conv: Conv3d) -> None:
                _init_truncated_normal(conv.weight, conv.bias, std=init_std)

            return Conv3dPatchFeatureExtractor(
                config.num_input_channels,
                config.model_dim,
                patch_3d_dims,
                init_fn=init_conv3d,
            )
        elif num_patch_dims == 2:
            patch_2d_dims = cast(tuple[int, int], config.patch_dims)

            def init_conv2d(conv: Conv2d) -> None:
                _init_truncated_normal(conv.weight, conv.bias, std=init_std)

            return Conv2dPatchFeatureExtractor(
                config.num_input_channels,
                config.model_dim,
                patch_2d_dims,
                init_fn=init_conv2d,
            )
        else:
            raise ValueError(
                f"The length of `config.patch_dims` must be 2 or 3, but is {num_patch_dims} instead."
            )

    def create_position_encoder(self) -> InterpolatedPositionEncoder:
        config = self._config

        if len(config.input_dims) != len(config.patch_dims):
            raise ValueError(
                f"The lengths of `config.input_dims` and `config.patch_dims` must be equal, but they are {len(config.input_dims)} and {len(config.patch_dims)} instead."
            )

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
            )
        elif num_input_dims == 2:
            input_2d_dims = cast(tuple[int, int], config.input_dims)
            patch_2d_dims = cast(tuple[int, int], config.patch_dims)

            h_input_dim, w_input_dim = input_2d_dims
            h_patch_dim, w_patch_dim = patch_2d_dims

            grid_2d_dims = (h_input_dim // h_patch_dim), (w_input_dim // w_patch_dim)

            return Sinusoidal2dPositionEncoder(config.model_dim, grid_2d_dims)
        else:
            raise ValueError(
                f"The length of `config.input_dims` must be 2 or 3, but is {num_input_dims} instead."
            )

    def create_encoder(self) -> TransformerEncoder:
        config = self._config

        layers = []

        for idx in range(config.num_encoder_layers):
            layer = self.create_encoder_layer(idx)

            layers.append(layer)

        layer_norm = self.create_layer_norm()

        return StandardTransformerEncoder(layers, layer_norm)

    def create_encoder_layer(self, layer_idx: int) -> TransformerEncoderLayer:
        config = self._config

        self_attn = self.create_self_attention(layer_idx)

        self_attn_layer_norm = self.create_layer_norm()

        ffn = self.create_ffn(layer_idx)

        ffn_layer_norm = self.create_layer_norm()

        drop_path = DropPathResidualConnect(drop_p=config.droppath_p)

        return StandardTransformerEncoderLayer(
            self_attn,
            self_attn_layer_norm,
            ffn,
            ffn_layer_norm,
            norm_order=TransformerNormOrder.PRE,
            self_attn_residual=drop_path,
            ffn_residual=drop_path,
            dropout_p=config.dropout_p,
        )

    def create_self_attention(self, layer_idx: int) -> MultiheadAttention:
        config = self._config

        attn_bias = IdentityBias()

        sdpa = create_default_sdpa(attn_bias, dropout_p=config.attn_dropout_p)

        output_proj = self.create_mha_output_projection(layer_idx)

        return StandardMultiheadAttention(
            config.model_dim,
            config.num_encoder_attn_heads,
            sdpa,
            bias=config.qkv_bias,
            output_proj=output_proj,
        )

    def create_mha_output_projection(self, layer_idx: int) -> Linear:
        config = self._config

        init_std = config.init_std

        def init_projection(proj: Linear) -> None:
            _init_truncated_normal(proj.weight, proj.bias, std=init_std)

            with torch.no_grad():
                proj.weight.div_((2.0 * (layer_idx + 1)) ** 0.5)

        return Linear(
            config.model_dim, config.model_dim, bias=True, init_fn=init_projection
        )

    def create_ffn(self, layer_idx: int) -> FeedForwardNetwork:
        config = self._config

        init_std = config.init_std

        def init_projection(proj: Linear) -> None:
            _init_truncated_normal(proj.weight, proj.bias, std=init_std)

            with torch.no_grad():
                proj.weight.div_((2.0 * (layer_idx + 1)) ** 0.5)

        inner_dim = int(config.model_dim * config.ffn_inner_dim_ratio)

        return StandardFeedForwardNetwork(
            config.model_dim,
            inner_dim,
            bias=True,
            inner_activation=GELU(),
            proj_init_fn=init_projection,
        )

    def create_layer_norm(self) -> LayerNorm:
        config = self._config

        init_std = config.init_std

        def init_layer_norm(m: StandardLayerNorm) -> None:
            if m.weight is not None:
                _init_truncated_normal(m.weight, m.bias, std=init_std)

        return StandardLayerNorm(
            config.model_dim, bias=True, eps=1e-6, init_fn=init_layer_norm
        )


def _init_truncated_normal(
    weight: Tensor, bias: Tensor | None, *, std: float = 1.0
) -> None:
    nn.init.trunc_normal_(weight, mean=0.0, std=std)

    if bias is not None:
        nn.init.zeros_(bias)
