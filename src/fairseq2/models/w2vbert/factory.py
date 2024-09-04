# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from fairseq2.config_registry import ConfigRegistry
from fairseq2.models.factory import model_factories
from fairseq2.models.w2vbert.model import W2VBertModel
from fairseq2.models.wav2vec2 import (
    Wav2Vec2Builder,
    Wav2Vec2Config,
    Wav2Vec2EncoderConfig,
)
from fairseq2.nn.transformer import TransformerNormOrder
from fairseq2.typing import DataType, Device

W2VBERT_FAMILY: Final = "w2vbert"


@dataclass(kw_only=True)
class W2VBertConfig:
    """Holds the configuration of a w2v-BERT model.

    The default values correspond to the base architecture as described in
    :cite:t`https://doi.org/10.48550/arxiv.2108.06209`.
    """

    w2v2_config: Wav2Vec2Config = field(
        default_factory=lambda: Wav2Vec2Config(
            encoder_config=Wav2Vec2EncoderConfig(
                model_dim=1024,
                max_seq_len=4096,
                feature_dim=160,
                use_fbank=True,
                first_pass_dropout_p=0.0,
                layer_norm_features=False,
                feature_extractor_layer_descs=[],
                feature_extractor_bias=False,
                feature_extractor_layer_norm_convs=False,
                feature_gradient_scale=0.0,
                num_fbank_channels=80,
                fbank_stride=2,
                sample_fbank_every_k=1,
                pos_encoder_type="relative",
                pos_encoder_depth=0,
                pos_conv_kernel_size=0,
                num_pos_conv_groups=0,
                use_conformer=True,
                num_encoder_layers=24,
                num_encoder_attn_heads=16,
                ffn_inner_dim=4096,
                dropout_p=0.0,
                attn_dropout_p=0.0,
                layer_drop_p=0.0,
                norm_order=TransformerNormOrder.POST,
                depthwise_conv_kernel_size=31,
            ),
            final_dim=768,
            final_proj_bias=True,
            temporal_mask_span_len=10,
            max_temporal_mask_prob=0.65,
            min_num_temporal_mask_spans=2,
            spatial_mask_span_len=10,
            max_spatial_mask_prob=0.0,
            min_num_spatial_mask_spans=2,
            quantized_dim=1024,
            num_codebooks=1,
            num_codebook_entries=1024,
            codebook_sampling_temperature=(2.0, 0.1, 0.999995),
            num_distractors=100,
            logit_temp=0.1,
        )
    )
    """The configuration of the wav2vec 2.0 model."""

    num_bert_encoder_layers: int = 16
    """The number of encoder layers to use for masked prediction."""

    num_target_codebooks: int = 1
    """The number of consecutive codebooks to use as masked prediction targets."""


w2vbert_archs = ConfigRegistry[W2VBertConfig]()

w2vbert_arch = w2vbert_archs.decorator


class W2VBertBuilder:
    """Builds modules of a w2v-BERT model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2108.06209`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    _config: W2VBertConfig
    _w2v2_builder: Wav2Vec2Builder
    _device: Device | None
    _dtype: DataType | None

    def __init__(
        self,
        config: W2VBertConfig,
        w2v2_builder: Wav2Vec2Builder | None = None,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param config:
            The configuration.
        :param w2v2_builder:
            The wav2vec 2.0 builder.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        encoder_config = config.w2v2_config.encoder_config

        if encoder_config.layer_drop_p != 0.0:
            raise ValueError("w2v-BERT does not support LayerDrop.")

        if config.num_bert_encoder_layers >= encoder_config.num_encoder_layers:
            raise ValueError(
                f"`config.num_bert_encoder_layers` must be less than `config.w2v2_config.encoder_config.num_encoder_layers` ({encoder_config.num_encoder_layers}), but is {config.num_bert_encoder_layers} instead."
            )

        if config.num_target_codebooks > config.w2v2_config.num_codebooks:
            raise ValueError(
                f"`config.num_target_codebooks` must be less than the number of codebooks ({config.w2v2_config.num_codebooks}), but is {config.num_target_codebooks} instead."
            )

        self._config = config

        if w2v2_builder is None:
            w2v2_builder = Wav2Vec2Builder(
                config.w2v2_config, device=device, dtype=dtype
            )

        self._w2v2_builder = w2v2_builder

        self._device, self._dtype = device, dtype

    def build_model(self) -> W2VBertModel:
        """Build a model."""
        w2v2_model = self._w2v2_builder.build_model()

        model = W2VBertModel(
            w2v2_model,
            self._config.num_bert_encoder_layers,
            num_target_codebooks=self._config.num_target_codebooks,
            device=self._device,
            dtype=self._dtype,
        )

        model.set_family(W2VBERT_FAMILY)

        return model


def create_w2vbert_model(
    config: W2VBertConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> W2VBertModel:
    """Create a w2v-BERT model."""
    return W2VBertBuilder(config, device=device, dtype=dtype).build_model()


model_factories.register(
    W2VBERT_FAMILY, create_w2vbert_model, W2VBertConfig, w2vbert_archs
)
