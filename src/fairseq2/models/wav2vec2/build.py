# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import AbstractSet, Final, List, Literal, Optional, Tuple

import torch
from torch.nn import GELU, SiLU

from fairseq2.models.conformer import ConformerBlock, ConformerConvolution
from fairseq2.models.feature_extractor import SequenceFeatureExtractor
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
    get_default_sdpa,
)


@dataclass
class Wav2Vec2Config:
    """Holds the configuration of a wav2vec 2.0 model.

    The default values correspond to the *base* architecture described in
    Section 4.2 of :cite:t:`baevski2020wav2vec`.
    """

    model_dim: int = 768
    """The dimensionality of the model."""

    max_seq_len: int = 1024
    """The expected maximum sequence length after feature extraction."""

    # Features
    feature_dim: int = 512
    """The dimensionality of extracted features."""

    use_fbank: bool = False
    """If ``True``, uses log-mel filterbanks instead of waveforms as input."""

    post_extract_dropout_p: float = 0.0
    """The dropout probability on extracted features before masking and
    positional encoding."""

    layer_norm_features: bool = True
    """If ``True``, applies Layer Normalization to extracted features."""

    # Waveform Feature Extractor
    feature_extractor_layer_descs: List[Tuple[int, int, int]] = field(
        # fmt: off
        default_factory=lambda: [(512, 10, 5)]
                              + [(512,  3, 2)] * 4
                              + [(512,  2, 2)]
                              + [(512,  2, 2)]
        # fmt: on
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
    num_fbank_channels: int = 80
    """The number of source log-mel filterbank channels."""

    fbank_stride: int = 2

    sample_fbank_every_k: int = 1

    # Mask
    temporal_mask_span_len: int = 10
    """The length of each temporal mask span that is applied over time steps."""

    max_temporal_mask_prob: float = 0.65
    """The maximum probability of masking a time step. Note that, due to mask
    span overlap, the effective probability might be smaller."""

    spatial_mask_span_len: int = 10
    """The length of each spatial mask span that is applied over features."""

    max_spatial_mask_prob: float = 0.0
    """The maximum probability of masking a feature. Note that, due to mask span
    overlap, the effective probability might be smaller."""

    # Position Encoder
    pos_encoder_type: Literal["conv", "relative", "rotary"] = "conv"
    """The type of position encoder."""

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
    """The number of Transformer encoder layers."""

    num_encoder_attn_heads: int = 12
    """The number of attention heads in Transformer encoder layers."""

    ffn_inner_dim: int = 3072
    """The dimensionality of inner projection layers in Transformer feed-forward
    networks."""

    final_dim: int = 256
    """The dimensionality of the final projection that is applied to context
    network outputs and quantized targets."""

    final_proj_bias: bool = True
    """If ``True``, the final projection learns an additive bias."""

    dropout_p: float = 0.1
    """The dropout probability in Transformer layers."""

    attn_dropout_p: float = 0.1
    """The dropout probability on Transformer attention weights."""

    layer_drop_p: float = 0.05
    """If greater than zero, applies LayerDrop to Transformer encoder layers
    as described in :cite:t:`https://doi.org/10.48550/arxiv.1909.11556`."""

    norm_order: TransformerNormOrder = TransformerNormOrder.POST
    """The Layer Normalization order."""

    depthwise_conv_kernel_size: int = 31
    """The kernel size of depthwise convolutions in Conformer blocks."""

    # Quantization
    quantized_dim: int = 256
    """The output dimensionality of vector quantizer."""

    num_latent_vars: int = 320
    """The number of latent variables in each group of the quantizer codebook."""

    num_latent_groups: int = 2
    """The number of groups of latent variables in the quantizer codebook."""

    latent_temperature: Tuple[float, float, float] = (2, 0.5, 0.999995)
    """A tuple of start temperature, end temperature, and decay factor for
    latent variable sampling."""

    # Loss
    num_distractors: int = 100
    """The number of distractors to use in contrastive prediction."""

    logit_temp: float = 0.1
    """The temperature to divide logits by."""

    diversity_loss_weight: float = 0.1
    """The weight of diversity in loss computation."""


_CONFIGS: Final = {"base": lambda: Wav2Vec2Config()}


def get_wav2vec2_archs() -> AbstractSet[str]:
    """Return the names of supported wav2vec 2.0 architectures."""
    return _CONFIGS.keys()


def get_wav2vec2_config(arch_name: str) -> Wav2Vec2Config:
    """Return the configuration of the specified wav2vec 2.0 architecture.

    :param arch_name:
        The name of the architecture.
    """
    try:
        return _CONFIGS[arch_name]()
    except KeyError:
        raise ValueError(
            f"`arch_name` must be a known wav2vec 2.0 architecture, but is '{arch_name}' instead."
        )


def create_wav2vec2_model(
    cfg: Wav2Vec2Config,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Wav2Vec2Model:
    """Create a wav2vec 2.0 model as described in :cite:t:`baevski2020wav2vec`.

    :param cfg:
        The configuration to use.
    :param device:
        The device on which to initialize the model.
    :param dtype:
        The data type of the model parameters and buffers.
    """
    return Wav2Vec2Builder(cfg, device, dtype).build_model()


class Wav2Vec2Builder:
    """Builds modules of a wav2vec 2.0 model as described in
    :cite:t:`baevski2020wav2vec`.

    To tweak the model architecture, you can derive from this class and override
    the corresponding methods.
    """

    cfg: Wav2Vec2Config
    cached_rel_pos_encoding: Optional[RelativePositionalEncoding]
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]

    def __init__(
        self,
        cfg: Wav2Vec2Config,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param cfg:
            The configuration to use.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        if cfg.use_conformer and cfg.norm_order != TransformerNormOrder.POST:
            raise ValueError(
                f"`cfg.norm_order` must be `POST` when `cfg.use_conformer` is `True`, but is {cfg.norm_order} instead."
            )

        self.cfg = cfg
        self.cached_rel_pos_encoding = None
        self.device = device
        self.dtype = dtype

    def build_model(self) -> Wav2Vec2Model:
        """Build a model."""
        encoder_frontend = self.build_encoder_frontend()

        encoder = self.build_encoder()

        masker = self.build_masker()

        quantizer = self.build_quantizer()

        return Wav2Vec2Model(
            encoder_frontend,
            encoder,
            masker,
            quantizer,
            self.cfg.final_dim,
            self.cfg.final_proj_bias,
            self.cfg.num_distractors,
            self.cfg.logit_temp,
            self.cfg.diversity_loss_weight,
            device=self.device,
            dtype=self.dtype,
        )

    def build_encoder_frontend(self) -> Wav2Vec2Frontend:
        """Build a wav2vec 2.0 Transformer encoder front-end."""
        feature_extractor = self.build_feature_extractor()

        pos_encoder = self.build_position_encoder()

        return Wav2Vec2Frontend(
            self.cfg.model_dim,
            self.cfg.feature_dim,
            feature_extractor,
            pos_encoder,
            first_pass_dropout_p=self.cfg.post_extract_dropout_p,
            layer_norm=self.cfg.layer_norm_features,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_feature_extractor(self) -> Optional[SequenceFeatureExtractor]:
        """Build a feature extractor."""
        if self.cfg.use_fbank:
            return Wav2Vec2FbankFeatureExtractor(
                self.cfg.num_fbank_channels,
                self.cfg.fbank_stride,
                self.cfg.sample_fbank_every_k,
            )

        return Wav2Vec2FeatureExtractor(
            self.cfg.feature_extractor_layer_descs,
            self.cfg.feature_extractor_bias,
            layer_norm=self.cfg.feature_extractor_layer_norm_convs,
            grad_scale=self.cfg.feature_grad_scale,
            device=self.device,
            dtype=self.dtype,
        )

    def build_masker(self) -> Wav2Vec2Masker:
        """Build a temporal/spatial feature masker."""
        return Wav2Vec2Masker(
            self.cfg.model_dim,
            self.cfg.temporal_mask_span_len,
            self.cfg.max_temporal_mask_prob,
            self.cfg.spatial_mask_span_len,
            self.cfg.max_spatial_mask_prob,
            device=self.device,
            dtype=self.dtype,
        )

    def build_position_encoder(self) -> Optional[PositionEncoder]:
        """Build a position encoder."""
        if self.cfg.pos_encoder_type != "conv":
            return None

        if self.cfg.pos_encoder_depth == 1:
            return Wav2Vec2PositionEncoder(
                self.cfg.model_dim,
                self.cfg.pos_conv_kernel_size,
                self.cfg.num_pos_conv_groups,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            return Wav2Vec2StackedPositionEncoder(
                self.cfg.model_dim,
                self.cfg.pos_conv_kernel_size,
                self.cfg.num_pos_conv_groups,
                self.cfg.pos_encoder_depth,
                device=self.device,
                dtype=self.dtype,
            )

    def build_encoder(self) -> TransformerEncoder:
        """Build a Transformer encoder."""
        layers = [
            self.build_encoder_layer() for _ in range(self.cfg.num_encoder_layers)
        ]

        return StandardTransformerEncoder(
            layers,
            layer_drop_p=self.cfg.layer_drop_p,
            norm_order=self.cfg.norm_order,
            device=self.device,
            dtype=self.dtype,
        )

    def build_encoder_layer(self) -> TransformerEncoderLayer:
        """Build a Transformer encoder layer."""
        if self.cfg.use_conformer:
            return self.build_conformer_block()

        self_attn = self.build_attention()

        ffn = self.build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.cfg.dropout_p,
            norm_order=self.cfg.norm_order,
            device=self.device,
            dtype=self.dtype,
        )

    def build_conformer_block(self) -> TransformerEncoderLayer:
        """Build a Conformer block."""
        ffn1 = self.build_ffn(use_swish=True)

        self_attn = self.build_attention()

        conv = ConformerConvolution(
            self.cfg.model_dim,
            self.cfg.depthwise_conv_kernel_size,
            device=self.device,
            dtype=self.dtype,
        )

        ffn2 = self.build_ffn(use_swish=True)

        return ConformerBlock(
            ffn1,
            self_attn,
            conv,
            ffn2,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_attention(self) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        if self.cfg.pos_encoder_type == "rotary":
            pos_encoder = RotaryEncoder(
                self.cfg.model_dim // self.cfg.num_encoder_attn_heads,
                self.cfg.max_seq_len,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            pos_encoder = None

        sdpa: SDPA

        if self.cfg.pos_encoder_type == "relative":
            if self.cached_rel_pos_encoding is None:
                self.cached_rel_pos_encoding = RelativePositionalEncoding(
                    self.cfg.model_dim, self.cfg.max_seq_len, self.device, self.dtype
                )

            sdpa = RelativePositionSDPA(
                self.cfg.model_dim,
                self.cfg.num_encoder_attn_heads,
                self.cached_rel_pos_encoding,
                attn_dropout_p=self.cfg.attn_dropout_p,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            sdpa = get_default_sdpa(self.cfg.attn_dropout_p)

        return StandardMultiheadAttention(
            self.cfg.model_dim,
            self.cfg.num_encoder_attn_heads,
            pos_encoder=pos_encoder,
            sdpa=sdpa,
            device=self.device,
            dtype=self.dtype,
        )

    def build_ffn(self, use_swish: bool = False) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self.cfg.model_dim,
            self.cfg.ffn_inner_dim,
            inner_activation=SiLU() if use_swish else GELU(),
            norm_order=self.cfg.norm_order,
            device=self.device,
            dtype=self.dtype,
        )

    def build_quantizer(self) -> VectorQuantizer:
        """Build a vector quantizer."""
        return GumbelVectorQuantizer(
            self.cfg.feature_dim,
            self.cfg.quantized_dim,
            self.cfg.num_latent_groups,
            self.cfg.num_latent_vars,
            self.cfg.latent_temperature,
            device=self.device,
            dtype=self.dtype,
        )
