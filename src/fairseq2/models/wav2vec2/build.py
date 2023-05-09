# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import AbstractSet, Final, List, Literal, Optional, Tuple

import torch
from torch.nn import GELU, SiLU

from fairseq2.models.conformer import ConformerConvolution, ConformerEncoderLayer
from fairseq2.models.sequence_feature_extractor import SequenceFeatureExtractor
from fairseq2.models.wav2vec2.feature_extractor import (
    Wav2Vec2FbankFeatureExtractor,
    Wav2Vec2FeatureExtractor,
)
from fairseq2.models.wav2vec2.feature_masker import Wav2Vec2FeatureMasker
from fairseq2.models.wav2vec2.frontend import Wav2Vec2Frontend
from fairseq2.models.wav2vec2.model import Wav2Vec2Model
from fairseq2.models.wav2vec2.positional_encoder import (
    Wav2Vec2PositionalEncoder,
    Wav2Vec2StackedPositionalEncoder,
)
from fairseq2.models.wav2vec2.vector_quantizer import GumbelVectorQuantizer
from fairseq2.nn.positional_encoder import PositionalEncoder
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
)


@dataclass
class Wav2Vec2Config:
    """Holds the configuration of a wav2vec 2.0 model.

    The default values correspond to the *base* architecture described in
    Section 4.2 of :cite:t:`baevski2020wav2vec`.
    """

    model_dim: int = 768
    """The dimensionality of the model."""

    embed_dim: int = 512

    input_type: Literal["waveform", "fbank"] = "waveform"

    # Feature Extractor
    feature_extractor_layers: List[Tuple[int, int, int]] = field(
        # fmt: off
        default_factory=lambda: [(512, 10, 5)]
                              + [(512,  3, 2)] * 4
                              + [(512,  2, 2)]
                              + [(512,  2, 2)]
        # fmt: on
    )
    """A tuple of output dimension, kernel size, and stride length for each
    feature extraction layer."""

    feature_extractor_bias: bool = False
    """If ``True``, convolutions in feature extraction layers learn an additive
    bias."""

    feature_extractor_use_layer_norm: bool = False

    feature_grad_scale: float = 0.1

    feature_dropout_p: float = 0.0

    # Fbank Feature Extractor
    num_fbank_features: int = 80

    fbank_stride: int = 2

    sample_fbank_every_k: int = 0

    # Mask
    temporal_mask_span_len: int = 10

    max_temporal_mask_prob: float = 0.65

    spatial_mask_span_len: int = 10

    max_spatial_mask_prob: float = 0.0

    # Positional Encoder
    pos_encoder_type: Literal["conv", "abs", "rotary"] = "conv"
    """The type of positional encoder.

    The default value `conv` means convolutional positional encoder as
    described in the paper. The other two values `abs` and `rotary` stand for
    sinusoidal positional encoder and rotary encoder respectively. They are
    typically used with Conformer blocks.
    """

    # Convolutional Positional Encoder
    pos_encoder_depth: int = 1
    """The number of stacked positional encoder layers."""

    pos_conv_kernel_size: int = 128
    """The total kernel size of 1D convolutions in positional encoder layers."""

    num_pos_conv_groups: int = 16
    """The number of convolution groups in positional encoder layers."""

    # Quantization
    latent_vars: int = 320

    latent_groups: int = 2

    latent_temp: Tuple[float, float, float] = (2, 0.5, 0.999995)

    # Encoder
    use_conformer: bool = False
    """If ``True``, uses Conformer blocks instead of Transformer encoder layers."""

    num_encoder_layers: int = 12
    """The number of encoder layers."""

    num_encoder_attn_heads: int = 12
    """The number of attention heads in encoder layers."""

    ffn_inner_dim: int = 3072
    """The dimensionality of inner projection layers in feed-forward networks."""

    final_dim: int = 256

    dropout_p: float = 0.1

    attn_dropout_p: float = 0.1

    layer_drop_p: float = 0.05

    norm_order: TransformerNormOrder = TransformerNormOrder.POST

    depthwise_conv_kernel_size: int = 31
    """The kernel size of depthwise convolutions in Conformer blocks."""

    # Logits
    num_negatives: int = 100

    logit_temp: float = 0.1

    diversity_weight: float = 0.1

    dtype: torch.dtype = torch.float32
    """The data type of model parameters and buffers."""


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
            f"`arch_name` must be a known S2T Transformer architecture, but is '{arch_name}' instead."
        )


def create_wav2vec2_model(
    cfg: Wav2Vec2Config,
    device: Optional[torch.device] = None,
) -> Wav2Vec2Model:
    """Create an wav2vec 2.0 model as described in :cite:t:`baevski2020wav2vec`.

    :param cfg:
        The configuration to use.
    :param device:
        The device on which to initialize the model.
    """
    return Wav2Vec2Builder(cfg, device).build_model()


class Wav2Vec2Builder:
    cfg: Wav2Vec2Config
    device: Optional[torch.device]

    def __init__(
        self, cfg: Wav2Vec2Config, device: Optional[torch.device] = None
    ) -> None:
        self.cfg = cfg
        self.device = device

    def build_model(self) -> Wav2Vec2Model:
        """Build a model."""
        encoder_frontend = self.build_encoder_frontend()

        encoder = self.build_encoder()

        vector_quantizer = self.build_vector_quantizer()

        return Wav2Vec2Model(
            encoder_frontend,
            encoder,
            vector_quantizer,
            self.cfg.final_dim,
            self.cfg.num_negatives,
            self.cfg.logit_temp,
            self.cfg.diversity_weight,
            self.device,
            self.cfg.dtype,
        )

    def build_encoder_frontend(self) -> Wav2Vec2Frontend:
        feature_extractor = self.build_feature_extractor()

        mask = self.build_mask()

        pos_encoder = self.build_positional_encoder()

        return Wav2Vec2Frontend(
            self.cfg.model_dim,
            feature_extractor,
            mask,
            pos_encoder,
            feature_dropout_p=self.cfg.feature_dropout_p,
            norm_order=self.cfg.norm_order,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_feature_extractor(self) -> SequenceFeatureExtractor:
        if self.cfg.input_type == "fbank":
            return Wav2Vec2FbankFeatureExtractor(
                self.cfg.num_fbank_features,
                self.cfg.fbank_stride,
                self.cfg.sample_fbank_every_k,
            )
        else:
            return Wav2Vec2FeatureExtractor(
                self.cfg.feature_extractor_layers,
                self.cfg.feature_extractor_bias,
                use_layer_norm=self.cfg.feature_extractor_use_layer_norm,
                grad_scale=self.cfg.feature_grad_scale,
                device=self.device,
                dtype=self.cfg.dtype,
            )

    def build_mask(self) -> Wav2Vec2FeatureMasker:
        return Wav2Vec2FeatureMasker(
            self.cfg.model_dim,
            self.cfg.temporal_mask_span_len,
            self.cfg.max_temporal_mask_prob,
            self.cfg.spatial_mask_span_len,
            self.cfg.max_spatial_mask_prob,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_positional_encoder(self) -> PositionalEncoder:
        #        if self.cfg.pos_embed_type == "conv":
        if self.cfg.pos_encoder_depth == 1:
            return Wav2Vec2PositionalEncoder(
                self.cfg.model_dim,
                self.cfg.pos_conv_kernel_size,
                self.cfg.num_pos_conv_groups,
                device=self.device,
                dtype=self.cfg.dtype,
            )
        else:
            return Wav2Vec2StackedPositionalEncoder(
                self.cfg.model_dim,
                self.cfg.pos_conv_kernel_size,
                self.cfg.num_pos_conv_groups,
                self.cfg.pos_encoder_depth,
                device=self.device,
                dtype=self.cfg.dtype,
            )

    def build_encoder(self) -> TransformerEncoder:
        layers = [
            self.build_encoder_layer() for _ in range(self.cfg.num_encoder_layers)
        ]

        # TODO: check in __init__
        if self.cfg.use_conformer:
            norm_order = self.cfg.norm_order
        else:
            # We do not apply Layer Normalization to the output of the encoder
            # since Conformer blocks already apply it.
            norm_order = TransformerNormOrder.POST

        return StandardTransformerEncoder(
            layers,
            layer_drop_p=self.cfg.layer_drop_p,
            norm_order=norm_order,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_encoder_layer(self) -> TransformerEncoderLayer:
        """Build an encoder layer."""
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
            dtype=self.cfg.dtype,
        )

    def build_conformer_block(self) -> TransformerEncoderLayer:
        """Build a Conformer block."""
        ffn1 = self.build_ffn(use_swish=True)

        self_attn = self.build_attention()

        conv = ConformerConvolution(
            self.cfg.model_dim,
            self.cfg.depthwise_conv_kernel_size,
            device=self.device,
            dtype=self.cfg.dtype,
        )

        ffn2 = self.build_ffn(use_swish=True)

        return ConformerEncoderLayer(
            ffn1,
            self_attn,
            conv,
            ffn2,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_attention(self) -> MultiheadAttention:
        """Build a multi-head attention layer."""
        return StandardMultiheadAttention(
            self.cfg.num_encoder_attn_heads,
            self.cfg.model_dim,
            attn_dropout_p=self.cfg.attn_dropout_p,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_ffn(self, use_swish: bool = False) -> FeedForwardNetwork:
        """Build a feed-forward network."""
        return StandardFeedForwardNetwork(
            self.cfg.model_dim,
            self.cfg.ffn_inner_dim,
            inner_activation=SiLU() if use_swish else GELU(),
            norm_order=self.cfg.norm_order,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_vector_quantizer(self) -> GumbelVectorQuantizer:
        return GumbelVectorQuantizer(
            dim=self.cfg.embed_dim,
            num_vars=self.cfg.latent_vars,
            temp=self.cfg.latent_temp,
            groups=self.cfg.latent_groups,
            combine_groups=False,
            vq_dim=self.cfg.final_dim,
            device=self.device,
        )
