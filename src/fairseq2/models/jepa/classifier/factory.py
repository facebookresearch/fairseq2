# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import final

from fairseq2.config_registry import ConfigRegistry
from fairseq2.models.factory import model_factories
from fairseq2.models.jepa import JepaEncoderBuilder, JepaEncoderConfig
from fairseq2.models.jepa.classifier.model import (
    AttentivePooler,
    CrossAttentionDecoderLayer,
    JepaClassifierModel,
)
from fairseq2.nn.projection import IdentityProjection, Linear, Projection
from fairseq2.nn.transformer import (
    MultiheadAttention,
    StandardMultiheadAttention,
    create_default_sdpa,
)
from fairseq2.typing import DataType, Device

JEPA_CLASSIFIER_FAMILY = "jepa_classifier"


@dataclass(kw_only=True)
class JepaClassifierConfig:
    encoder_config: JepaEncoderConfig = field(
        default_factory=lambda: JepaEncoderConfig()
    )
    """The configuration of the vision encoder."""

    pool_depth: int = 1
    """The pool depth (minimum 1 decoder layer)"""

    decoder_projection: bool = True
    """If True, the decoder will have a linear layer on top"""

    num_queries: int = 1
    """Number of query tokens in the attention pool layer"""

    num_classes: int = 1000
    """Size of classification logits"""


jepa_classifier_archs = ConfigRegistry[JepaClassifierConfig]()

jepa_classifier_arch = jepa_classifier_archs.decorator


@final
class JepaClassifierBuilder:
    """Build a JEPA model fine-tuned for classification"""

    _config: JepaClassifierConfig
    _encoder_builder: JepaEncoderBuilder
    _device: Device | None
    _dtype: DataType | None

    def __init__(
        self,
        config: JepaClassifierConfig,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        self._config = config

        self._encoder_builder = JepaEncoderBuilder(
            config.encoder_config, device=device, dtype=dtype
        )

        self._device, self._dtype = device, dtype

    def build_model(self) -> JepaClassifierModel:
        encoder_frontend = self._encoder_builder.build_frontend()
        encoder = self._encoder_builder.build_encoder()
        pooler = self.build_pooler()
        head = self.build_head()

        return JepaClassifierModel(encoder_frontend, encoder, pooler, head)

    def build_pooler(self) -> AttentivePooler:
        config = self._config

        if config.pool_depth > 1:
            encoder = self._encoder_builder.build_encoder(config.pool_depth)
        else:
            encoder = None

        decoder = self.build_decoder_layer()

        return AttentivePooler(
            decoder=decoder,
            encoder=encoder,
            num_queries=config.num_queries,
            init_std=config.encoder_config.init_std,
            device=self._device,
            dtype=self._dtype,
        )

    def build_head(self) -> Projection:
        config = self._config
        return Linear(
            config.encoder_config.model_dim,
            config.num_classes,
            device=self._device,
            dtype=self._dtype,
            bias=True,
        )

    def build_decoder_layer(self) -> CrossAttentionDecoderLayer:
        config = self._config

        cross_attn = self.build_cross_attention()

        ffn = self._encoder_builder.build_ffn(config.pool_depth)

        return CrossAttentionDecoderLayer(
            cross_attn,
            ffn,
            layer_norm_factory=self._encoder_builder.build_layer_norm,
            device=self._device,
            dtype=self._dtype,
        )

    def build_cross_attention(self) -> MultiheadAttention:
        config = self._config.encoder_config

        model_dim = config.model_dim

        sdpa = create_default_sdpa(attn_dropout_p=config.attn_dropout_p)

        output_proj = self.build_cross_attn_output_projection()

        return StandardMultiheadAttention(
            model_dim,
            config.num_encoder_attn_heads,
            sdpa=sdpa,
            bias=config.qkv_bias,
            output_proj=output_proj,
            device=self._device,
            dtype=self._dtype,
        )

    def build_cross_attn_output_projection(self) -> Projection:
        config = self._config

        model_dim = config.encoder_config.model_dim

        if config.decoder_projection:
            return Linear(
                model_dim,
                model_dim,
                bias=True,
                device=self._device,
                dtype=self._dtype,
            )
        else:
            return IdentityProjection(model_dim, model_dim)


def create_jepa_classifier_model(
    config: JepaClassifierConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> JepaClassifierModel:
    return JepaClassifierBuilder(
        config,
        device=device,
        dtype=dtype,
    ).build_model()


model_factories.register(
    JEPA_CLASSIFIER_FAMILY,
    create_jepa_classifier_model,
    JepaClassifierConfig,
    jepa_classifier_archs,
)
