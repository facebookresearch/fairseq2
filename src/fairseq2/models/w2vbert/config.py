# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from fairseq2.models.transformer import TransformerNormOrder
from fairseq2.models.wav2vec2 import Wav2Vec2Config, Wav2Vec2EncoderConfig
from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.runtime.dependency import DependencyContainer
from fairseq2.utils.validation import ValidationResult

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
                feature_grad_scale=0.0,
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

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        encoder_config = self.w2v2_config.encoder_config

        if encoder_config.layer_drop_p != 0.0:
            result.add_error(
                f"`w2v2_config.encoder_config.layer_drop_p` must be 0.0 since w2v-BERT does not support LayerDrop, but is {encoder_config.layer_drop_p} instead."
            )

        if self.num_bert_encoder_layers >= encoder_config.num_encoder_layers:
            result.add_error(
                f"`num_bert_encoder_layers` must be less than `w2v2_config.encoder_config.num_encoder_layers` ({encoder_config.num_encoder_layers}), but is {self.num_bert_encoder_layers} instead."
            )

        if self.num_target_codebooks > self.w2v2_config.num_codebooks:
            result.add_error(
                f"`num_target_codebooks` must be less than `w2v2_config.num_codebooks` ({self.w2v2_config.num_codebooks}), but is {self.num_target_codebooks} instead."
            )

        return result


def register_w2vbert_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, W2VBertConfig)

    @arch("300m")
    def _300m() -> W2VBertConfig:
        config = _600m()

        config.w2v2_config.encoder_config.num_encoder_layers = 12

        config.num_bert_encoder_layers = 8

        return config

    @arch("600m")
    def _600m() -> W2VBertConfig:
        return W2VBertConfig()
