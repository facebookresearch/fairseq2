# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
import math
from typing import Final, cast

import torch
from torch.nn import GELU, Module

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
    """std to initialize the weights and bias for linear and LayerNorm layers"""

    dropout_p: float = 0.0
    """The dropout probability on outputs of Transformer layers."""

    droppath_p: float = 0.0
    """The probability of output sequence drop."""

    uniform_power: bool = False
    """
    If ``True``, each patch dimension will have equal representation in the
    produced positional encodings.
    """


jepa_archs = ConfigRegistry[JepaConfig]()

jepa_arch = jepa_archs.decorator


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
        
        conv_init_fn = partial(init_with_explicit_bounds, std=config.init_std)

        num_patch_dims = len(config.patch_dims)

        if num_patch_dims == 3:
            patch_3d_dims = cast(tuple[int, int, int], config.patch_dims)

            return Conv3dPatchFeatureExtractor(
                config.num_input_channels,
                config.model_dim,
                patch_3d_dims,
                init_fn=conv_init_fn,
                device=self._device,
                dtype=self._dtype,
            )
        elif num_patch_dims == 2:
            patch_2d_dims = cast(tuple[int, int], config.patch_dims)

            return Conv2dPatchFeatureExtractor(
                config.num_input_channels,
                config.model_dim,
                patch_2d_dims,
                init_fn=conv_init_fn,
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

    def build_encoder(self) -> TransformerEncoder:
        config = self._config

        num_layers = config.num_encoder_layers

        layers = [self.build_encoder_layer(i) for i in range(num_layers)]

        return StandardTransformerEncoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.build_layer_norm,
            device=self._device,
            dtype=self._dtype,
        )

    def build_encoder_layer(self, layer_id: int) -> TransformerEncoderLayer:
        config = self._config

        self_attn = self.build_attention(layer_id)

        ffn = self.build_ffn(layer_id)
        
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

    def build_attention(self, layer_id: int) -> MultiheadAttention:
        config = self._config

        sdpa = create_default_sdpa(attn_dropout_p=config.attn_dropout_p)

        proj = self.build_projection(layer_id)

        return StandardMultiheadAttention(
            config.model_dim,
            config.num_encoder_attn_heads,
            sdpa=sdpa,
            bias=config.qkv_bias,
            output_proj=proj,
            output_proj_bias=True,
            device=self._device,
            dtype=self._dtype,
        )

    def build_projection(self, layer_id: int) -> Linear:
        config = self._config

        proj_init_fn: Callable[[Linear], None] = partial(
            init_with_explicit_bounds, std=config.init_std
        )

        proj = Linear(
            config.model_dim,
            config.model_dim,
            bias=True,
            init_fn=proj_init_fn,
            device=self._device,
            dtype=self._dtype,
        )

        # rescale the linear layer
        proj.weight.data.div_(math.sqrt(2.0 * layer_id))

        return proj

    def build_ffn(self, layer_id: int) -> FeedForwardNetwork:
        config = self._config

        proj_init_fn = partial(init_with_explicit_bounds, std=config.init_std)

        ffn = StandardFeedForwardNetwork(
            config.model_dim,
            int(config.model_dim * config.ffn_inner_dim_ratio),
            bias=True,
            inner_activation=GELU(),
            proj_init_fn=proj_init_fn,
            norm_order=TransformerNormOrder.PRE,
            device=self._device,
            dtype=self._dtype,
        )

        # rescale the last layer
        proj = ffn.output_proj
        assert isinstance(proj, Linear), f"Invalid projection type: {type(proj)}"
        proj.weight.data.div_(math.sqrt(2.0 * layer_id))
        
        return ffn

    def build_layer_norm(
        self,
        model_dim: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> LayerNorm:
        config = self._config
        
        layer_norm_init_fn = partial(init_with_explicit_bounds, std=config.init_std)
        
        return StandardLayerNorm(
            model_dim, bias=True, eps=1e-6, init_fn=layer_norm_init_fn, device=device, dtype=dtype
        )


def create_jepa_model(
    config: JepaConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> JepaModel:
    return JepaBuilder(config, device=device, dtype=dtype).build_model()


def _norm_cdf(x):
    # Computes standard normal cumulative distribution function
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    

def normalize_truncate(
    tensor: torch.Tensor,
    *,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> None:

    lower = _norm_cdf((a - mean) / std)
    upper = _norm_cdf((b - mean) / std)
    
    tensor.uniform_(2 * lower - 1, 2 * upper - 1)
    tensor.erfinv_()
    
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)
    
    tensor.clamp_(min=a, max=b)
        

def init_with_explicit_bounds(
    m: Module,
    *,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
):
    if not hasattr(m, "weight") or not hasattr(m, "bias"):
        raise ValueError(f"Cannot initialize weights and bias of a {type(m)}")
    
    with torch.no_grad():
        normalize_truncate(m.weight, mean=mean, std=std, a=a, b=b)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
