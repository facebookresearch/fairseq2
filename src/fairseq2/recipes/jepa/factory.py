# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from functools import partial
from typing import final

import torch
from torch import Tensor
from torch.nn import GELU

from fairseq2.config_registry import ConfigRegistry
from fairseq2.models.jepa.factory import init_truncated_normal
from fairseq2.nn.normalization import LayerNorm, StandardLayerNorm
from fairseq2.nn.projection import IdentityProjection, Linear, Projection
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    StandardMultiheadAttention,
    create_default_sdpa,
)
from fairseq2.nn.transformer.encoder import (
    StandardTransformerEncoder,
    TransformerEncoder,
)
from fairseq2.nn.transformer.encoder_layer import (
    StandardTransformerEncoderLayer,
    TransformerEncoderLayer,
)
from fairseq2.nn.transformer.ffn import StandardFeedForwardNetwork
from fairseq2.nn.transformer.norm_order import TransformerNormOrder
from fairseq2.recipes.jepa.models import (
    AttentiveClassifier,
    AttentivePooler,
    CrossAttentionDecoder,
)
from fairseq2.typing import DataType, Device


@dataclass(kw_only=True)
class AttentivePoolerConfig:
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

    pool_depth: int = 1
    """The pool depth (minimum 1 decoder layer)"""

    num_attn_heads: int = 12
    """The number of attention heads in encoder layers."""

    qkv_bias: bool = True
    """
    If ``True``, query, key, and value projections in multi-head attention
    layers will have an additive bias.
    """

    num_queries: int = 1
    """Number of query tokens in the attention pool layer"""

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


@dataclass(kw_only=True)
class AttentiveClassifierConfig:

    pooler_config: AttentivePoolerConfig = field(
        default_factory=lambda: AttentivePoolerConfig()
    )

    num_classes: int = 1000
    """Size of classification logits"""


attentive_archs = ConfigRegistry[AttentiveClassifierConfig]()

attentive_arch = attentive_archs.decorator


@final
class AttentiveClassifierBuilder:
    """Build a Jepa model that is fine-tuned for classification"""

    _config: AttentiveClassifierConfig
    _device: Device | None
    _dtype: DataType | None

    def __init__(
        self,
        config: AttentiveClassifierConfig,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:

        self._config = config

        self.pooler_builer = AttentivePoolerBuilder(
            config.pooler_config, device=device, dtype=dtype
        )

        self._device, self._dtype = device, dtype

    def build_model(self) -> AttentiveClassifier:
        config = self._config

        pooler = self.pooler_builer.build_model()

        head = Linear(config.pooler_config.model_dim, config.num_classes, bias=True)

        return AttentiveClassifier(pooler, head)


@final
class AttentivePoolerBuilder:
    """
    Build an attentive pooler. Many builer functions are similar to JepaEncoderBuilder
    since we have an optional transformer encoder in the pool

    TODO: Refactor to have common building blocks for jepa encoder and pooler
    in a base builder class (?)
    """

    _config: AttentivePoolerConfig
    _device: Device | None
    _dtype: DataType | None

    def __init__(
        self,
        config: AttentivePoolerConfig,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        self._config = config

        self._device, self._dtype = device, dtype

    def build_model(self) -> AttentivePooler:
        config = self._config

        def init_pool(pool: Tensor) -> None:
            std = config.init_std
            with torch.no_grad():
                torch.nn.init.trunc_normal_(pool, std=std)

        decoder = self.build_decoder()

        if config.pool_depth > 1:
            encoder = self.build_encoder()
        else:
            encoder = None

        return AttentivePooler(
            decoder=decoder,
            encoder=encoder,
            num_queries=config.num_queries,
            init_fn=init_pool,
            device=self._device,
            dtype=self._dtype,
        )

    def build_decoder(self) -> CrossAttentionDecoder:
        config = self._config

        cross_attn = self.build_attention(config.pool_depth, is_cross_attn=True)

        ffn = self.build_ffn(config.pool_depth)

        return CrossAttentionDecoder(
            cross_attn,
            ffn,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.build_layer_norm,
            device=self._device,
            dtype=self._dtype,
        )

    def build_encoder(self) -> TransformerEncoder:
        config = self._config

        num_layers = config.pool_depth

        layers = [self.build_encoder_layer(i) for i in range(1, num_layers)]

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

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.build_layer_norm,
            device=self._device,
            dtype=self._dtype,
        )

    def build_attention(
        self, layer_idx: int, is_cross_attn: bool = False
    ) -> MultiheadAttention:
        config = self._config

        sdpa = create_default_sdpa(attn_dropout_p=config.attn_dropout_p)

        if is_cross_attn:
            output_proj: Projection = IdentityProjection(config.model_dim, config.model_dim)
        output_proj = self.build_mha_output_projection(layer_idx)

        return StandardMultiheadAttention(
            config.model_dim,
            config.num_attn_heads,
            sdpa=sdpa,
            bias=config.qkv_bias,
            output_proj=output_proj,
            device=self._device,
            dtype=self._dtype,
        )

    def build_mha_output_projection(self, layer_idx: int) -> Projection:
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
                proj.weight.div_(math.sqrt(2.0 * (layer_idx)))

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

        init_layer_norm = partial(
            init_truncated_normal, std=init_std
        )

        return StandardLayerNorm(
            model_dim,
            bias=True,
            eps=1e-6,
            init_fn=init_layer_norm,
            device=device,
            dtype=dtype,
        )


def create_attentive_pooler(
    config: AttentivePoolerConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> AttentivePooler:
    return AttentivePoolerBuilder(config, device=device, dtype=dtype).build_model()
