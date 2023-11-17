# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

from fairseq2.models.utils import ArchitectureRegistry
from fairseq2.models.w2vbert.model import W2VBertModel
from fairseq2.models.wav2vec2 import Wav2Vec2Config, Wav2Vec2EncoderConfig
from fairseq2.models.wav2vec2.builder import Wav2Vec2Builder, Wav2Vec2EncoderBuilder
from fairseq2.nn.transformer import TransformerNormOrder
from fairseq2.typing import DataType, Device


def _encoder_600m() -> Wav2Vec2EncoderConfig:
    return Wav2Vec2EncoderConfig(
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
    )


def _encoder_300m() -> Wav2Vec2EncoderConfig:
    return Wav2Vec2EncoderConfig(
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
        num_encoder_layers=12,
        num_encoder_attn_heads=16,
        ffn_inner_dim=4096,
        dropout_p=0.0,
        attn_dropout_p=0.0,
        layer_drop_p=0.0,
        norm_order=TransformerNormOrder.POST,
        depthwise_conv_kernel_size=31,
    )


@dataclass
class W2VBertConfig:
    """Holds the configuration of a w2v-BERT model."""

    w2v2_config: Wav2Vec2Config
    """The configuration of the wav2vec 2.0 model."""

    num_bert_encoder_layers: int
    """The number of Transformer encoder layers to use for masked prediction."""

    num_target_codebooks: int
    """The number of consecutive codebooks to use as masked prediction targets."""

    w2v2_loss_weight: float
    """The weight of wav2vec 2.0 loss in loss computation."""

    bert_loss_weight: float
    """The weight of masked prediction loss in loss computation."""

    bert_label_smoothing: float
    """The amount of label smoothing when computing masked prediction loss."""


w2vbert_archs = ArchitectureRegistry[W2VBertConfig]("w2v-bert")

w2vbert_arch = w2vbert_archs.decorator


@w2vbert_arch("600m")
def _600m() -> W2VBertConfig:
    w2v2_encoder_config = _encoder_600m()

    w2v2_config = Wav2Vec2Config(
        w2v2_encoder_config,
        final_dim=768,
        final_proj_bias=True,
        temporal_mask_span_len=10,
        max_temporal_mask_prob=0.65,
        spatial_mask_span_len=10,
        max_spatial_mask_prob=0.0,
        quantized_dim=1024,
        num_codebooks=1,
        num_codebook_entries=1024,
        codebook_sampling_temperature=(2.0, 0.1, 0.999995),
        num_distractors=100,
        logit_temp=0.1,
        diversity_loss_weight=0.2,
    )

    return W2VBertConfig(
        w2v2_config,
        num_bert_encoder_layers=16,
        num_target_codebooks=1,
        w2v2_loss_weight=1.0,
        bert_loss_weight=1.0,
        bert_label_smoothing=0.0,
    )


@w2vbert_arch("300m")
def _300m() -> W2VBertConfig:
    w2v2_encoder_config = _encoder_300m()

    w2v2_config = Wav2Vec2Config(
        w2v2_encoder_config,
        final_dim=768,
        final_proj_bias=True,
        temporal_mask_span_len=10,
        max_temporal_mask_prob=0.65,
        spatial_mask_span_len=10,
        max_spatial_mask_prob=0.0,
        quantized_dim=1024,
        num_codebooks=1,
        num_codebook_entries=1024,
        codebook_sampling_temperature=(2.0, 0.1, 0.999995),
        num_distractors=100,
        logit_temp=0.1,
        diversity_loss_weight=0.2,
    )

    return W2VBertConfig(
        w2v2_config,
        num_bert_encoder_layers=8,
        num_target_codebooks=1,
        w2v2_loss_weight=1.0,
        bert_loss_weight=1.0,
        bert_label_smoothing=0.0,
    )


class W2VBertBuilder:
    """Builds modules of a w2v-BERT model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2108.06209`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: W2VBertConfig
    w2v2_builder: Wav2Vec2Builder
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: W2VBertConfig,
        w2v2_builder: Wav2Vec2Builder,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
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
        encoder_builder = w2v2_builder.encoder_builder

        if encoder_builder.config.layer_drop_p != 0.0:
            raise ValueError("w2v-BERT does not support LayerDrop.")

        if config.num_bert_encoder_layers >= encoder_builder.config.num_encoder_layers:
            raise ValueError(
                f"`config.num_bert_encoder_layers` must be less than the number of Transformer encoder layers ({encoder_builder.config.num_encoder_layers}), but is {config.num_bert_encoder_layers} instead."
            )

        if config.num_target_codebooks > w2v2_builder.config.num_codebooks:
            raise ValueError(
                f"`config.num_target_codebooks` must be less than the number of codebooks ({w2v2_builder.config.num_codebooks}), but is {config.num_target_codebooks} instead."
            )

        self.config = config

        self.w2v2_builder = w2v2_builder

        self.device, self.dtype = device, dtype

    def build_model(self) -> W2VBertModel:
        """Build a model."""
        w2v2_model = self.w2v2_builder.build_model()

        return W2VBertModel(
            w2v2_model,
            self.config.num_bert_encoder_layers,
            num_target_codebooks=self.config.num_target_codebooks,
            w2v2_loss_weight=self.config.w2v2_loss_weight,
            bert_loss_weight=self.config.bert_loss_weight,
            bert_label_smoothing=self.config.bert_label_smoothing,
            device=self.device,
            dtype=self.dtype,
        )


def create_w2vbert_model(
    config: W2VBertConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> W2VBertModel:
    """Create a w2v-BERT model.

    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    encoder_builder = Wav2Vec2EncoderBuilder(
        config.w2v2_config.encoder_config, device=device, dtype=dtype
    )

    w2v2_builder = Wav2Vec2Builder(
        config.w2v2_config, encoder_builder, device=device, dtype=dtype
    )

    builder = W2VBertBuilder(config, w2v2_builder, device=device, dtype=dtype)

    return builder.build_model()
