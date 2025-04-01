# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from fairseq2.context import RuntimeContext
from fairseq2.nn.transformer import TransformerNormOrder
from fairseq2.utils.validation import ValidationError, ValidationResult

WAV2VEC2_MODEL_FAMILY: Final = "wav2vec2"


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
    """The dimensionality of the final projection that is applied to context
    network outputs and quantized targets."""

    final_proj_bias: bool = True
    """If ``True``, the final projection learns an additive bias."""

    quantizer_encoder_grad: bool = True
    """If ``True``, gradients are propagated from the quantizer through the convolutional
    encoder. Otherwise, they are detached and the encoder is only trained with gradients
    from the transformer. """

    # Mask
    mask_codebase: str = "fairseq2"

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

    feature_gradient_scale: float = 0.1
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

    # Encoder (i.e. Context Network)
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

    def validate(self) -> None:
        result = ValidationResult()

        if self.use_conformer and self.norm_order != TransformerNormOrder.POST:
            result.add_error(
                f"`norm_order` must be `POST` when `use_conformer` is `True`, but is `{self.norm_order}` instead."
            )

        if result.has_error:
            raise ValidationError(
                "The wav2vec 2.0 encoder configuration has one or more validation errors:", result  # fmt: skip
            )


def register_wav2vec2_configs(context: RuntimeContext) -> None:
    arch = context.get_config_registry(Wav2Vec2Config).decorator
    arch_encoder = context.get_config_registry(Wav2Vec2EncoderConfig).decorator

    @arch("base")
    def base() -> Wav2Vec2Config:
        return Wav2Vec2Config()

    @arch_encoder("base")
    def base_encoder() -> Wav2Vec2EncoderConfig:
        return base().encoder_config

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

    @arch_encoder("large")
    def large_encoder() -> Wav2Vec2EncoderConfig:
        return large().encoder_config

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

    @arch_encoder("large_lv60k")
    def large_lv60k_encoder() -> Wav2Vec2EncoderConfig:
        return large_lv60k().encoder_config

    @arch("xlsr_base")
    def xlsr_base() -> Wav2Vec2Config:
        config = large_lv60k()
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.feature_gradient_scale = 1.0
        return config

    @arch_encoder("xlsr_base")
    def xlsr_base_encoder() -> Wav2Vec2EncoderConfig:
        return xlsr_base().encoder_config

    @arch("base_conformer")
    def base_conformer() -> Wav2Vec2Config:
        config = xlsr_base()

        config.encoder_config.use_conformer = True
        config.encoder_config.norm_order = TransformerNormOrder.POST
        config.encoder_config.depthwise_conv_kernel_size = 31
        # pos_encoder_type

        return config

    @arch_encoder("base_conformer")
    def base_conformer_encoder() -> Wav2Vec2EncoderConfig:
        return base_conformer().encoder_config

    @arch("1b")
    def b1() -> Wav2Vec2Config:
        config = xlsr_base()

        config.encoder_config.model_dim = 1280
        config.encoder_config.num_encoder_layers = 48
        config.encoder_config.ffn_inner_dim = 5120
        config.encoder_config.dropout_p = 0.0
        config.quantized_dim = 1024
        config.final_dim = 1024
        config.encoder_config.first_pass_dropout_p = 0.1

        return config

    @arch_encoder("1b")
    def b1_encoder() -> Wav2Vec2EncoderConfig:
        return b1().encoder_config

    @arch("2b")
    def b2() -> Wav2Vec2Config:
        config = b1()

        config.encoder_config.model_dim = 1920
        config.encoder_config.ffn_inner_dim = 7680

        return config

    @arch_encoder("2b")
    def b2_encoder() -> Wav2Vec2EncoderConfig:
        return b2().encoder_config


    @arch("3b")
    def b3() -> Wav2Vec2Config:
        config = b1()

        config.encoder_config.num_encoder_layers = 60
        config.encoder_config.model_dim = 2048
        config.encoder_config.ffn_inner_dim = 8192

        return config

    @arch_encoder("3b")
    def b3_encoder() -> Wav2Vec2EncoderConfig:
        return b3().encoder_config

    @arch("3b_mel")
    def mel_3b() -> Wav2Vec2Config:
        config = b3()

        config.encoder_config.use_fbank = True
        config.encoder_config.num_fbank_channels = 80
        config.encoder_config.fbank_stride = 2
        config.encoder_config.sample_fbank_every_k = 1
        config.encoder_config.feature_dim = 160

        return config

    @arch_encoder("3b_mel")
    def mel_3b_encoder() -> Wav2Vec2EncoderConfig:
        return mel_3b().encoder_config

    @arch("3.25b")
    def higher_3b() -> Wav2Vec2Config:
        config = b1()

        config.encoder_config.num_encoder_layers = 64
        config.encoder_config.model_dim = 2048
        config.encoder_config.ffn_inner_dim = 8192
        config.encoder_config.num_encoder_attn_heads = 32
        config.quantized_dim = 1280
        config.final_dim = 1280

        return config

    @arch_encoder("3.25b")
    def higher_3b_encoder() -> Wav2Vec2EncoderConfig:
        return higher_3b().encoder_config

    @arch("4b")
    def b4() -> Wav2Vec2Config:
        config = b2()

        config.quantized_dim = 1280
        config.final_dim = 1280
        config.encoder_config.num_encoder_layers = 64
        config.encoder_config.model_dim = 2304
        config.encoder_config.ffn_inner_dim = 9216
        config.encoder_config.num_encoder_attn_heads = 32

        return config

    @arch_encoder("4b")
    def b4_encoder() -> Wav2Vec2EncoderConfig:
        return b4().encoder_config

    @arch("1b_llama")
    def llama_1b() -> Wav2Vec2Config:
        config = xlsr_base()

        config.encoder_config.model_dim = 2048
        config.encoder_config.num_encoder_layers = 16
        config.encoder_config.ffn_inner_dim = int(2048 * 4 * 1.5)
        config.encoder_config.num_encoder_attn_heads = 32
        config.encoder_config.dropout_p = 0.0
        config.quantized_dim = 1024
        config.final_dim = 1024
        config.encoder_config.first_pass_dropout_p = 0.1

        return config

    @arch_encoder("1b_llama")
    def llama_1b_encoder() -> Wav2Vec2EncoderConfig:
        return llama_1b().encoder_config

    @arch("3b_llama")
    def llama_3b() -> Wav2Vec2Config:
        config = llama_1b()

        config.encoder_config.model_dim = 2560
        config.encoder_config.num_encoder_layers = 32
        config.encoder_config.ffn_inner_dim = int(2560 * 4 * 1.0)
        config.quantized_dim = 2048
        config.final_dim = 2048

        return config

    @arch_encoder("3b_llama")
    def llama_3b_encoder() -> Wav2Vec2EncoderConfig:
        return llama_3b().encoder_config

    @arch("5b")
    def b5() -> Wav2Vec2Config:
        config = b3()

        config.encoder_config.num_encoder_layers = 96
        config.encoder_config.model_dim = 2048
        config.encoder_config.ffn_inner_dim = 8192
        config.encoder_config.num_encoder_attn_heads = 16
        config.quantized_dim = 1024
        config.final_dim = 1024

        return config

    @arch_encoder("5b")
    def b5_encoder() -> Wav2Vec2EncoderConfig:
        return b5().encoder_config

    @arch("7b")
    def b7() -> Wav2Vec2Config:
        config = b5()

        config.encoder_config.num_encoder_layers = 128
        config.encoder_config.model_dim = 2048
        config.encoder_config.ffn_inner_dim = 8192
        config.encoder_config.num_encoder_attn_heads = 16
        config.quantized_dim = 1024
        config.final_dim = 1024

        return config

    @arch_encoder("7b")
    def b7_encoder() -> Wav2Vec2EncoderConfig:
        return b7().encoder_config

    @arch("pseudo_dinosr_base")
    def pseudo_dinosr_base() -> Wav2Vec2Config:
        layer_descs = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 3

        encoder_config = Wav2Vec2EncoderConfig(
            model_dim=768,
            max_seq_len=100000,
            feature_dim=512,
            use_fbank=False,
            first_pass_dropout_p=0.0,
            layer_norm_features=True,
            feature_extractor_layer_descs=layer_descs,
            feature_extractor_bias=False,
            feature_extractor_layer_norm_convs=True,
            feature_gradient_scale=0.1,
            num_fbank_channels=0,
            fbank_stride=0,
            sample_fbank_every_k=0,
            pos_encoder_type="conv",
            pos_encoder_depth=5,
            pos_conv_kernel_size=95,
            num_pos_conv_groups=16,
            use_conformer=False,
            num_encoder_layers=12,
            num_encoder_attn_heads=12,
            ffn_inner_dim=3072,
            dropout_p=0.1,
            attn_dropout_p=0.1,
            layer_drop_p=0.0,
            norm_order=TransformerNormOrder.POST,
            depthwise_conv_kernel_size=31,
        )

        return Wav2Vec2Config(
            encoder_config=encoder_config,
            final_dim=256,
            final_proj_bias=True,
            temporal_mask_span_len=10,
            max_temporal_mask_prob=0.65,
            spatial_mask_span_len=10,
            max_spatial_mask_prob=0.0,
            quantized_dim=256,
            num_codebooks=2,
            num_codebook_entries=320,
            codebook_sampling_temperature=(2.0, 0.5, 0.999995),
            num_distractors=100,
            logit_temp=0.1,
        )

    @arch_encoder("pseudo_dinosr_base")
    def pseudo_dinosr_base_encoder() -> Wav2Vec2EncoderConfig:
        return pseudo_dinosr_base().encoder_config
