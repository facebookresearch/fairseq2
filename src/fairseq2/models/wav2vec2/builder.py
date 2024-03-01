# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional, Tuple

from torch.nn import GELU, SiLU

from fairseq2.models.conformer import ConformerBlock, ConformerConvolution
from fairseq2.models.feature_extractor import SequenceFeatureExtractor
from fairseq2.models.utils import ArchitectureRegistry
from fairseq2.models.wav2vec2.feature_extractor import (
    Wav2Vec2FbankFeatureExtractor,
    Wav2Vec2FeatureExtractor,
)
from fairseq2.models.wav2vec2.frontend import Wav2Vec2Frontend
from fairseq2.models.wav2vec2.masker import Wav2Vec2Masker
from fairseq2.models.wav2vec2.model import Wav2Vec2Model
from fairseq2.models.wav2vec2.position_encoder import (
    Wav2Vec2PositionEncoder,
    Wav2Vec2StackedPositionEncoder,
)
from fairseq2.models.wav2vec2.vector_quantizer import (
    GumbelVectorQuantizer,
    VectorQuantizer,
)
from fairseq2.nn.position_encoder import PositionEncoder, RotaryEncoder
from fairseq2.nn.transformer import (
    SDPA,
    FeedForwardNetwork,
    MultiheadAttention,
    RelativePositionalEncoding,
    RelativePositionSDPA,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.typing import DataType, Device


@dataclass
class Wav2Vec2EncoderConfig:
    """Holds the configuration of a wav2vec 2.0 encoder."""

    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The maximum allowed sequence length after feature extraction."""

    # Features
    feature_dim: int
    """The dimensionality of extracted features."""

    use_fbank: bool
    """If ``True``, uses log-mel filterbanks instead of waveforms as input."""

    first_pass_dropout_p: float
    """The dropout probability on extracted features before masking and
    positional encoding."""

    layer_norm_features: bool
    """If ``True``, applies Layer Normalization to extracted features."""

    # Waveform Feature Extractor
    feature_extractor_layer_descs: List[Tuple[int, int, int]]
    """A tuple of output dimension, kernel size, and stride for each feature
    extraction layer."""

    feature_extractor_bias: bool
    """If ``True``, convolutions in feature extraction layers learn an additive
    bias."""

    feature_extractor_layer_norm_convs: bool
    """If ``True``, applies Layer Normalization to outputs of convolutions in
    feature extraction layers."""

    feature_grad_scale: float
    """The scale factor for gradients of extracted features. Setting to a value
    less than 1.0 allows the feature extractor to learn at a lower rate than the
    rest of the model."""

    # Filterbank Feature Extractor
    num_fbank_channels: int
    """The number of source log-mel filterbank channels."""

    fbank_stride: int

    sample_fbank_every_k: int

    # Position Encoder
    pos_encoder_type: str
    """The type of position encoder ('conv', 'relative', 'rotary')."""

    # Convolutional Position Encoder
    pos_encoder_depth: int
    """The number of stacked position encoder layers."""

    pos_conv_kernel_size: int
    """The total kernel size of 1D convolutions in position encoder layers."""

    num_pos_conv_groups: int
    """The number of convolution groups in position encoder layers."""

    # Encoder (i.e. Context Network)
    use_conformer: bool
    """If ``True``, uses Conformer blocks instead of Transformer encoder layers."""

    num_encoder_layers: int
    """The number of Transformer encoder layers."""

    num_encoder_attn_heads: int
    """The number of attention heads in Transformer encoder layers."""

    ffn_inner_dim: int
    """The inner dimensionality of Transformer feed-forward networks."""

    dropout_p: float
    """The dropout probability in Transformer layers."""

    attn_dropout_p: float
    """The dropout probability on Transformer attention weights."""

    layer_drop_p: float
    """If greater than zero, applies LayerDrop to Transformer encoder layers
    as described in :cite:t:`https://doi.org/10.48550/arxiv.1909.11556`."""

    norm_order: TransformerNormOrder
    """The Layer Normalization order."""

    depthwise_conv_kernel_size: int
    """The kernel size of depthwise convolutions in Conformer blocks."""


def _encoder_base() -> Wav2Vec2EncoderConfig:
    layer_descs = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    return Wav2Vec2EncoderConfig(
        model_dim=768,
        max_seq_len=4096,
        feature_dim=512,
        use_fbank=False,
        first_pass_dropout_p=0.0,
        layer_norm_features=True,
        feature_extractor_layer_descs=layer_descs,
        feature_extractor_bias=False,
        feature_extractor_layer_norm_convs=False,
        feature_grad_scale=0.1,
        num_fbank_channels=0,
        fbank_stride=0,
        sample_fbank_every_k=0,
        pos_encoder_type="conv",
        pos_encoder_depth=1,
        pos_conv_kernel_size=128,
        num_pos_conv_groups=16,
        use_conformer=False,
        num_encoder_layers=12,
        num_encoder_attn_heads=12,
        ffn_inner_dim=3072,
        dropout_p=0.1,
        attn_dropout_p=0.1,
        layer_drop_p=0.05,
        norm_order=TransformerNormOrder.POST,
        depthwise_conv_kernel_size=0,
    )


class Wav2Vec2EncoderBuilder:
    """Builds modules of a wav2vec 2.0 encoder as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    _config: Wav2Vec2EncoderConfig
    _device: Optional[Device]
    _dtype: Optional[DataType]
    _rel_pos_encoding: Optional[RelativePositionalEncoding]

    def __init__(
        self,
        config: Wav2Vec2EncoderConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        if config.use_conformer and config.norm_order != TransformerNormOrder.POST:
            raise ValueError(
                f"`config.norm_order` must be `POST` when `config.use_conformer` is `True`, but is `{config.norm_order}` instead."
            )

        self._config = config

        self._device, self._dtype = device, dtype

        self._rel_pos_encoding = None

    def build_frontend(self) -> Wav2Vec2Frontend:
        """Build a wav2vec 2.0 Transformer encoder front-end."""
        feature_extractor = self.build_feature_extractor()

        pos_encoder = self.build_position_encoder()

        return Wav2Vec2Frontend(
            self._config.model_dim,
            self._config.feature_dim,
            feature_extractor,
            pos_encoder,
            first_pass_dropout_p=self._config.first_pass_dropout_p,
            layer_norm=self._config.layer_norm_features,
            dropout_p=self._config.dropout_p,
            device=self._device,
            dtype=self._dtype,
        )

    def build_feature_extractor(self) -> Optional[SequenceFeatureExtractor]:
        """Build a feature extractor."""
        if self._config.use_fbank:
            return Wav2Vec2FbankFeatureExtractor(
                self._config.num_fbank_channels,
                self._config.fbank_stride,
                sample_every_k=self._config.sample_fbank_every_k,
            )

        return Wav2Vec2FeatureExtractor(
            self._config.feature_extractor_layer_descs,
            self._config.feature_extractor_bias,
            layer_norm=self._config.feature_extractor_layer_norm_convs,
            grad_scale=self._config.feature_grad_scale,
            device=self._device,
            dtype=self._dtype,
        )

    def build_position_encoder(self) -> Optional[PositionEncoder]:
        """Build a position encoder."""
        if self._config.pos_encoder_type != "conv":
            return None

        if self._config.pos_encoder_depth == 1:
            return Wav2Vec2PositionEncoder(
                self._config.model_dim,
                self._config.pos_conv_kernel_size,
                self._config.num_pos_conv_groups,
                device=self._device,
                dtype=self._dtype,
            )
        else:
            return Wav2Vec2StackedPositionEncoder(
                self._config.model_dim,
                self._config.pos_conv_kernel_size,
                self._config.num_pos_conv_groups,
                self._config.pos_encoder_depth,
                device=self._device,
                dtype=self._dtype,
            )

    def build_encoder(self) -> TransformerEncoder:
        """Build a Transformer encoder."""
        num_layers = self._config.num_encoder_layers

        layers = [self.build_encoder_layer() for _ in range(num_layers)]

        return StandardTransformerEncoder(
            layers,
            layer_drop_p=self._config.layer_drop_p,
            norm_order=self._config.norm_order,
            device=self._device,
            dtype=self._dtype,
        )

    def build_encoder_layer(self) -> TransformerEncoderLayer:
        """Build a Transformer encoder layer."""
        if self._config.use_conformer:
            return self.build_conformer_block()

        self_attn = self.build_attention()

        ffn = self.build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self._config.dropout_p,
            norm_order=self._config.norm_order,
            device=self._device,
            dtype=self._dtype,
        )

    def build_conformer_block(self) -> TransformerEncoderLayer:
        """Build a Conformer block."""
        ffn1 = self.build_ffn(use_swish=True)

        self_attn = self.build_attention()

        conv = self.build_conformer_conv()

        ffn2 = self.build_ffn(use_swish=True)

        return ConformerBlock(
            ffn1,
            self_attn,
            conv,
            ffn2,
            dropout_p=self._config.dropout_p,
            device=self._device,
            dtype=self._dtype,
        )

    def build_attention(self) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        if self._config.pos_encoder_type == "rotary":
            pos_encoder = RotaryEncoder(
                self._config.model_dim // self._config.num_encoder_attn_heads,
                self._config.max_seq_len,
                device=self._device,
            )
        else:
            pos_encoder = None

        sdpa = self.build_sdpa()

        return StandardMultiheadAttention(
            self._config.model_dim,
            self._config.num_encoder_attn_heads,
            pos_encoder=pos_encoder,
            sdpa=sdpa,
            device=self._device,
            dtype=self._dtype,
        )

    def build_sdpa(self) -> SDPA:
        sdpa = create_default_sdpa(attn_dropout_p=self._config.attn_dropout_p)

        if self._config.pos_encoder_type == "relative":
            if self._rel_pos_encoding is None:
                self._rel_pos_encoding = RelativePositionalEncoding(
                    self._config.model_dim,
                    self._config.max_seq_len,
                    device=self._device,
                    dtype=self._dtype,
                )

            sdpa = RelativePositionSDPA(
                self._config.model_dim,
                self._config.num_encoder_attn_heads,
                self._rel_pos_encoding,
                inner_sdpa=sdpa,
                device=self._device,
                dtype=self._dtype,
            )

        return sdpa

    def build_conformer_conv(self) -> ConformerConvolution:
        return ConformerConvolution(
            self._config.model_dim,
            self._config.depthwise_conv_kernel_size,
            device=self._device,
            dtype=self._dtype,
        )

    def build_ffn(self, use_swish: bool = False) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self._config.model_dim,
            self._config.ffn_inner_dim,
            bias=True,
            inner_activation=SiLU() if use_swish else GELU(),
            norm_order=self._config.norm_order,
            device=self._device,
            dtype=self._dtype,
        )


@dataclass
class Wav2Vec2Config:
    """Holds the configuration of a wav2vec 2.0 model."""

    encoder_config: Wav2Vec2EncoderConfig
    """The configuration of the wav2vec 2.0 encoder."""

    final_dim: int
    """The dimensionality of the final projection that is applied to context
    network outputs and quantized targets."""

    final_proj_bias: bool
    """If ``True``, the final projection learns an additive bias."""

    # Mask
    temporal_mask_span_len: int
    """The length of each temporal mask span that is applied over time steps."""

    max_temporal_mask_prob: float
    """The maximum probability of masking a time step. Note that, due to mask
    span overlap, the effective probability might be smaller."""

    min_num_temporal_mask_spans: int
    """The minimum number of temporal mask sampled per sequence."""

    spatial_mask_span_len: int
    """The length of each spatial mask span that is applied over features."""

    max_spatial_mask_prob: float
    """The maximum probability of masking a feature. Note that, due to mask span
    overlap, the effective probability might be smaller."""

    min_num_spatial_mask_spans: int
    """The minimum number of spatial mask sampled per sequence."""

    # Quantization
    quantized_dim: int
    """The output dimensionality of vector quantizer."""

    num_codebooks: int
    """The number of codebooks."""

    num_codebook_entries: int
    """The number of entries per codebook."""

    codebook_sampling_temperature: Tuple[float, float, float]
    """A tuple of start temperature, end temperature, and decay factor for
    codebook entry sampling."""

    # Loss
    num_distractors: int
    """The number of distractors to use in contrastive prediction."""

    logit_temp: float
    """The temperature to divide logits by."""

    diversity_loss_weight: float
    """The weight of diversity in loss computation."""


wav2vec2_archs = ArchitectureRegistry[Wav2Vec2Config]("wav2vec2")

wav2vec2_arch = wav2vec2_archs.decorator


@wav2vec2_arch("base")
def _base() -> Wav2Vec2Config:
    encoder_config = _encoder_base()

    return Wav2Vec2Config(
        encoder_config,
        final_dim=256,
        final_proj_bias=True,
        temporal_mask_span_len=10,
        max_temporal_mask_prob=0.65,
        min_num_temporal_mask_spans=2,
        spatial_mask_span_len=10,
        max_spatial_mask_prob=0.0,
        min_num_spatial_mask_spans=2,
        quantized_dim=256,
        num_codebooks=2,
        num_codebook_entries=320,
        codebook_sampling_temperature=(2, 0.5, 0.999995),
        num_distractors=100,
        logit_temp=0.1,
        diversity_loss_weight=0.1,
    )


class Wav2Vec2Builder:
    """Builds modules of a wav2vec 2.0 model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    _config: Wav2Vec2Config
    _encoder_builder: Wav2Vec2EncoderBuilder
    _device: Optional[Device]
    _dtype: Optional[DataType]

    def __init__(
        self,
        config: Wav2Vec2Config,
        encoder_builder: Wav2Vec2EncoderBuilder,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration.
        :param encoder_builder_cls:
            The wav2vec 2.0 encoder builder.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self._config = config

        self._encoder_builder = encoder_builder

        self._device, self._dtype = device, dtype

    def build_model(self) -> Wav2Vec2Model:
        """Build a model."""
        encoder_frontend = self._encoder_builder.build_frontend()

        encoder = self._encoder_builder.build_encoder()

        masker = self.build_masker()

        quantizer = self.build_quantizer()

        return Wav2Vec2Model(
            encoder_frontend,
            encoder,
            masker,
            quantizer,
            self._config.final_dim,
            final_proj_bias=self._config.final_proj_bias,
            num_distractors=self._config.num_distractors,
            logit_temp=self._config.logit_temp,
            diversity_loss_weight=self._config.diversity_loss_weight,
            device=self._device,
            dtype=self._dtype,
        )

    def build_masker(self) -> Wav2Vec2Masker:
        """Build a temporal/spatial feature masker."""
        return Wav2Vec2Masker(
            self._config.encoder_config.model_dim,
            self._config.temporal_mask_span_len,
            self._config.max_temporal_mask_prob,
            self._config.min_num_temporal_mask_spans,
            self._config.spatial_mask_span_len,
            self._config.max_spatial_mask_prob,
            self._config.min_num_spatial_mask_spans,
            device=self._device,
            dtype=self._dtype,
        )

    def build_quantizer(self) -> VectorQuantizer:
        """Build a vector quantizer."""
        return GumbelVectorQuantizer(
            self._config.encoder_config.feature_dim,
            self._config.quantized_dim,
            self._config.num_codebooks,
            self._config.num_codebook_entries,
            codebook_sampling_temperature=self._config.codebook_sampling_temperature,
            device=self._device,
            dtype=self._dtype,
        )


def create_wav2vec2_model(
    config: Wav2Vec2Config,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> Wav2Vec2Model:
    """Create a wav2vec 2.0 model.

    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    encoder_builder = Wav2Vec2EncoderBuilder(
        config.encoder_config, device=device, dtype=dtype
    )

    builder = Wav2Vec2Builder(config, encoder_builder, device=device, dtype=dtype)

    return builder.build_model()
