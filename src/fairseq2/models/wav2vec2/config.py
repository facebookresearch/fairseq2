# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from fairseq2.models.transformer import TransformerNormOrder
from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.runtime.dependency import DependencyContainer
from fairseq2.utils.validation import ValidationResult

WAV2VEC2_FAMILY: Final = "wav2vec2"


@dataclass(kw_only=True)
class Wav2Vec2Config:
    """Holds the configuration of a wav2vec 2.0 model.

    The default values correspond to the base architecture as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.
    """

    encoder_config: Wav2Vec2EncoderConfig = field(
        default_factory=lambda: Wav2Vec2EncoderConfig()
    )
    """The configuration of the wav2vec 2.0 encoder."""

    final_dim: int = 256
    """The dimensionality of the final projection that is applied to resolver
    network outputs and quantized targets."""

    final_proj_bias: bool = True
    """If ``True``, the final projection learns an additive bias."""

    quantizer_encoder_grad: bool = True
    """If ``True``, gradients are propagated from the quantizer through the convolutional
    encoder. Otherwise, they are detached and the encoder is only trained with gradients
    from the transformer. """

    # Mask
    temporal_mask_span_len: int = 10
    """The length of each temporal mask span that is applied over time steps."""

    max_temporal_mask_prob: float = 0.69
    """The maximum probability of masking a time step. Note that, due to mask
    span overlap, the effective probability will be lower."""

    min_num_temporal_mask_spans: int = 2
    """The minimum number of temporal masks sampled per sequence."""

    spatial_mask_span_len: int = 10
    """The length of each spatial mask span that is applied over features."""

    max_spatial_mask_prob: float = 0.0
    """The maximum probability of masking a feature. Note that, due to mask span
    overlap, the effective probability will be lower."""

    min_num_spatial_mask_spans: int = 2
    """The minimum number of spatial masks sampled per sequence."""

    # Quantization
    quantized_dim: int = 256
    """The output dimensionality of vector quantizer."""

    num_codebooks: int = 2
    """The number of codebooks."""

    num_codebook_entries: int = 320
    """The number of entries per codebook."""

    codebook_sampling_temperature: tuple[float, float, float] = (2.0, 0.5, 0.999995)
    """A tuple of start temperature, end temperature, and decay factor for
    codebook entry sampling."""

    # Loss
    num_distractors: int = 100
    """The number of distractors to use in contrastive prediction."""

    logit_temp: float = 0.1
    """The temperature to divide logits by."""


@dataclass(kw_only=True)
class Wav2Vec2EncoderConfig:
    """Holds the configuration of a wav2vec 2.0 encoder.

    The default values correspond to the base architecture described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.
    """

    model_dim: int = 768
    """The dimensionality of the model."""

    max_seq_len: int = 4096
    """The maximum sequence length after feature extraction."""

    # Features
    feature_dim: int = 512
    """The dimensionality of extracted features."""

    use_fbank: bool = False
    """If ``True``, uses log-mel filterbanks instead of waveforms as input."""

    first_pass_dropout_p: float = 0.0
    """The dropout probability on extracted features before masking and
    positional encoding."""

    layer_norm_features: bool = True
    """If ``True``, applies Layer Normalization to extracted features."""

    # Waveform Feature Extractor
    feature_extractor_layer_descs: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
    )
    """A tuple of output dimension, kernel size, and stride for each feature
    extraction layer."""

    feature_extractor_bias: bool = False
    """If ``True``, convolutions in feature extraction layers learn an additive
    bias."""

    feature_extractor_layer_norm_convs: bool = False
    """If ``True``, applies Layer Normalization to outputs of convolutions in
    feature extraction layers."""

    feature_grad_scale: float = 0.1
    """The scale factor for gradients of extracted features. Setting to a value
    less than 1.0 allows the feature extractor to learn at a lower rate than the
    rest of the model."""

    # Filterbank Feature Extractor
    num_fbank_channels: int = 0
    """The number of source log-mel filterbank channels."""

    fbank_stride: int = 0

    sample_fbank_every_k: int = 0

    # Position Encoder
    pos_encoder_type: str = "conv"
    """The type of position encoder ('conv', 'relative', 'rotary')."""

    # Convolutional Position Encoder
    pos_encoder_depth: int = 1
    """The number of stacked position encoder layers."""

    pos_conv_kernel_size: int = 128
    """The total kernel size of 1D convolutions in position encoder layers."""

    num_pos_conv_groups: int = 16
    """The number of convolution groups in position encoder layers."""

    # Encoder (i.e. resolver Network)
    use_conformer: bool = False
    """If ``True``, uses Conformer blocks instead of Transformer encoder layers."""

    num_encoder_layers: int = 12
    """The number of encoder layers."""

    num_encoder_attn_heads: int = 12
    """The number of attention heads in encoder layers."""

    ffn_inner_dim: int = 3072
    """The inner dimensionality of feed-forward networks."""

    dropout_p: float = 0.1
    """The dropout probability on outputs of Transformer layers."""

    attn_dropout_p: float = 0.1
    """The dropout probability on attention weights."""

    ffn_inner_dropout_p: float = 0.0
    """The dropout probability on inner activations of feed-forward networks."""

    layer_drop_p: float = 0.05
    """If greater than zero, applies LayerDrop to encoder layers as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1909.11556`."""

    norm_order: TransformerNormOrder = TransformerNormOrder.POST
    """The Layer Normalization order."""

    depthwise_conv_kernel_size: int = 0
    """The kernel size of depthwise convolutions in Conformer blocks."""

    def validate(self) -> ValidationResult:
        result = ValidationResult()

        if self.use_conformer and self.norm_order != TransformerNormOrder.POST:
            result.add_error(
                f"`norm_order` must be `POST` when `use_conformer` is `True`, but is `{self.norm_order}` instead."
            )

        return result


def register_wav2vec2_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, Wav2Vec2Config)

    @arch("base")
    def base() -> Wav2Vec2Config:
        return Wav2Vec2Config()

    @arch("large")
    def large() -> Wav2Vec2Config:
        config = base()

        config.encoder_config.model_dim = 1024
        config.encoder_config.num_encoder_layers = 24
        config.encoder_config.num_encoder_attn_heads = 16
        config.encoder_config.ffn_inner_dim = 4096
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.layer_drop_p = 0.2
        config.quantized_dim = 768
        config.final_dim = 768

        return config

    @arch("large_lv60k")
    def large_lv60k() -> Wav2Vec2Config:
        config = large()

        config.encoder_config.layer_norm_features = False
        config.encoder_config.feature_extractor_bias = True
        config.encoder_config.feature_extractor_layer_norm_convs = True
        config.encoder_config.layer_drop_p = 0.0
        config.encoder_config.norm_order = TransformerNormOrder.PRE
        config.codebook_sampling_temperature = (2.0, 0.1, 0.999995)

        return config
