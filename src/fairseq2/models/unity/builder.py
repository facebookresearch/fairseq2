# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

from fairseq2.models.conformer import ConformerBlock, ConformerConvolution
from fairseq2.models.transformer import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    TransformerModel,
)
from fairseq2.models.unity.adaptor_block import (
    UnitYConformerAdaptorLayer,
    UnitYS2TEncoderAdaptor,
    UnitYTransformerAdaptorLayer,
)
from fairseq2.models.unity.model import UnitYModel
from fairseq2.models.utils.arch import ArchitectureRegistry
from fairseq2.models.w2vbert import w2vbert_archs
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderBuilder, Wav2Vec2EncoderConfig
from fairseq2.nn.embedding import Embedding
from fairseq2.nn.position_encoder import SinusoidalPositionEncoder
from fairseq2.nn.projection import TiedProjection
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerNormOrder,
    get_default_sdpa,
)
from fairseq2.typing import DataType, Device


@dataclass
class UnitYS2TConfig:
    """Holds the configuration of an S2T UnitY model."""

    model_dim: int
    """The dimensionality of the model."""

    target_max_seq_len: int
    """The expected maximum target sequence length."""

    target_vocabulary_size: int
    """The size of the target vocabulary."""

    target_pad_idx: Optional[int]
    """The index of the pad symbol in the target vocabulary."""

    w2v2_encoder_config: Wav2Vec2EncoderConfig
    """The configuration of the wav2vec 2.0 encoder."""

    use_conformer_adaptor: bool
    """If ``True``, uses a Conformer-based adaptor block."""

    num_adaptor_layers: int
    """The number of Transformer encoder layers in the adaptor block."""

    adaptor_kernel_size: int
    """The kernel size of 1D convolutions in the adaptor block."""

    adaptor_stride: int
    """The stride of 1D convolutions in the adaptor block."""

    num_decoder_layers: int
    """The number of Transformer decoder layers."""

    num_decoder_attn_heads: int
    """The number of attention heads in Transformer decoder layers."""

    ffn_inner_dim: int
    """The inner dimensionality of Transformer feed-forward networks."""

    dropout_p: float
    """The dropout probability in Transformer layers."""


unity_s2t_archs = ArchitectureRegistry[UnitYS2TConfig]("unity_s2t")


unity_s2t_arch = unity_s2t_archs.marker


@unity_s2t_arch("base")
def _base_s2t() -> UnitYS2TConfig:
    w2vbert_config = w2vbert_archs.get_config("600m")

    return UnitYS2TConfig(
        model_dim=1024,
        target_max_seq_len=2048,
        target_vocabulary_size=256206,  # NLLB-200
        target_pad_idx=0,
        w2v2_encoder_config=w2vbert_config.w2v2_config.encoder_config,
        num_decoder_layers=24,
        num_decoder_attn_heads=16,
        use_conformer_adaptor=True,
        num_adaptor_layers=1,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        ffn_inner_dim=1024 * 8,
        dropout_p=0.1,
    )


class UnitYS2TBuilder:
    """Builds modules of an S2T UnitY model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2212.08055`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: UnitYS2TConfig
    w2v2_encoder_builder: Wav2Vec2EncoderBuilder
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: UnitYS2TConfig,
        w2v2_encoder_builder: Wav2Vec2EncoderBuilder,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param w2v2_encoder_builder:
            The wav2vec 2.0 encoder builder.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        if config.model_dim != w2v2_encoder_builder.config.model_dim:
            raise ValueError(
                f"`model_dim` and `model_dim` of `w2v2_encoder_builder.config` must be equal, but are {config.model_dim} and {w2v2_encoder_builder.config.model_dim} instead."
            )

        self.config = config
        self.w2v2_encoder_builder = w2v2_encoder_builder
        self.device = device
        self.dtype = dtype

    def reset(self) -> None:
        """Reset the internal state of the builder."""

    def build_model(self) -> TransformerModel:
        """Build a model."""
        encoder_frontend = self.build_encoder_frontend()

        encoder = self.build_encoder()

        embed = Embedding(
            num_embeddings=self.config.target_vocabulary_size,
            embedding_dim=self.config.model_dim,
            pad_idx=self.config.target_pad_idx,
            scaled=True,
            device=self.device,
            dtype=self.dtype,
        )

        decoder_frontend = self.build_decoder_frontend(embed)

        decoder = self.build_decoder()

        final_proj = TiedProjection(embed.weight)

        return TransformerModel(
            encoder_frontend,
            encoder,
            decoder_frontend,
            decoder,
            final_proj,
            self.config.target_pad_idx,
        )

    def build_encoder_frontend(self) -> TransformerFrontend:
        """Build a Transformer encoder front-end."""
        return self.w2v2_encoder_builder.build_frontend()

    def build_decoder_frontend(self, embed: Embedding) -> TransformerFrontend:
        """Build a Transformer decoder front-end."""
        pos_encoder = SinusoidalPositionEncoder(
            self.config.model_dim,
            self.config.target_max_seq_len,
            _legacy_pad_idx=self.config.target_pad_idx,
            device=self.device,
            dtype=self.dtype,
        )

        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_encoder(self) -> TransformerEncoder:
        """Build a Transformer encoder."""
        w2v2_encoder = self.w2v2_encoder_builder.build_encoder()

        # For Conformer-based wav2vec 2.0 architectures (e.g. w2v-BERT), we
        # typically use a special type of adapter layer.
        if self.config.use_conformer_adaptor:
            build_adaptor_layer = self.build_conformer_adaptor_layer
        else:
            build_adaptor_layer = self.build_adaptor_layer

        num_layers = self.config.num_adaptor_layers

        layers = [build_adaptor_layer(i) for i in range(num_layers)]

        return UnitYS2TEncoderAdaptor(w2v2_encoder, layers, self.device, self.dtype)

    def build_adaptor_layer(self, idx: int) -> TransformerEncoderLayer:
        """Build a Transformer-based encoder adaptor layer."""
        self_attn = self.build_attention(
            self.w2v2_encoder_builder.config.num_encoder_attn_heads
        )

        ffn = self.w2v2_encoder_builder.build_ffn()

        return UnitYTransformerAdaptorLayer(
            self_attn,
            ffn,
            self.config.adaptor_kernel_size,
            self.config.adaptor_stride,
            self.config.dropout_p,
            self.device,
            self.dtype,
        )

    def build_conformer_adaptor_layer(self, idx: int) -> TransformerEncoderLayer:
        """Build a Conformer-based encoder adaptor layer."""
        ffn1 = self.w2v2_encoder_builder.build_ffn(use_swish=True)

        # Empirically shown that, in adaptor layers, vanilla MHA performs better
        # than MHA with relative positional encoding.
        self_attn = self.build_attention(
            self.w2v2_encoder_builder.config.num_encoder_attn_heads
        )

        conv = ConformerConvolution(
            self.w2v2_encoder_builder.config.model_dim,
            self.w2v2_encoder_builder.config.depthwise_conv_kernel_size,
            device=self.device,
            dtype=self.dtype,
        )

        ffn2 = self.w2v2_encoder_builder.build_ffn(use_swish=True)

        block = ConformerBlock(
            ffn1,
            self_attn,
            conv,
            ffn2,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

        layer_norm = idx == 0

        return UnitYConformerAdaptorLayer(
            block,
            self.config.adaptor_kernel_size,
            self.config.adaptor_stride,
            layer_norm,
            self.device,
            self.dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        num_layers = self.config.num_decoder_layers

        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        return StandardTransformerDecoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        self_attn = self.build_attention(self.config.num_decoder_attn_heads)

        encoder_decoder_attn = self.build_attention(self.config.num_decoder_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn,
            ffn,
            dropout_p=self.config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        sdpa = get_default_sdpa(attn_dropout_p=self.config.dropout_p)

        return StandardMultiheadAttention(
            self.config.model_dim,
            num_heads,
            sdpa=sdpa,
            device=self.device,
            dtype=self.dtype,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )


def create_unity_s2t_model(
    config: UnitYS2TConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> TransformerModel:
    """Create an S2T UnitY model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    w2v2_encoder_builder = Wav2Vec2EncoderBuilder(
        config.w2v2_encoder_config, device, dtype
    )

    return UnitYS2TBuilder(config, w2v2_encoder_builder, device, dtype).build_model()


@dataclass
class UnitYConfig:
    """Holds the configuration of a UnitY model."""

    unit_max_seq_len: int
    """The expected maximum unit sequence length."""

    unit_vocabulary_size: int
    """The size of the unit vocabulary."""

    unit_pad_idx: Optional[int]
    """The index of the pad symbol in the unit vocabulary."""

    s2t_model_config: UnitYS2TConfig
    """The configuration of the S2T UnitY model."""

    num_t2u_encoder_layers: int
    """The number of T2U Transformer encoder layers."""

    num_t2u_decoder_layers: int
    """The number of T2U Transformer decoder layers."""

    num_t2u_encoder_attn_heads: int
    """The number of attention heads in T2U Transformer encoder layers."""

    num_t2u_decoder_attn_heads: int
    """The number of attention heads in T2U Transformer decoder layers."""


unity_archs = ArchitectureRegistry[UnitYConfig]("unity")


unity_arch = unity_archs.marker


@unity_arch("base")
def _base() -> UnitYConfig:
    s2t_model_config = unity_s2t_archs.get_config("base")

    return UnitYConfig(
        unit_max_seq_len=1024,
        unit_vocabulary_size=1026,
        unit_pad_idx=0,
        s2t_model_config=s2t_model_config,
        # TODO: LAYERDROP??
        num_t2u_encoder_layers=6,
        num_t2u_decoder_layers=6,
        num_t2u_encoder_attn_heads=16,
        num_t2u_decoder_attn_heads=16,
    )


class UnitYBuilder:
    config: UnitYConfig
    s2t_model_builder: UnitYS2TBuilder
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: UnitYConfig,
        s2t_model_builder: UnitYS2TBuilder,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param s2t_model_builder:
            The S2T UnitY model builder.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config
        self.s2t_model_builder = s2t_model_builder
        self.device = device
        self.dtype = dtype

    def reset(self) -> None:
        """Reset the internal state of the builder."""

    def build_model(self) -> UnitYModel:
        """Build a model."""
        s2t_model = self.s2t_model_builder.build_model()

        unit_embed = Embedding(
            num_embeddings=self.config.unit_vocabulary_size,
            embedding_dim=self.config.s2t_model_config.model_dim,
            pad_idx=self.config.unit_pad_idx,
            scaled=True,
            device=self.device,
            dtype=self.dtype,
        )

        t2u_encoder = self.build_t2u_encoder()

        t2u_decoder_frontend = self.build_t2u_decoder_frontend(unit_embed)

        t2u_decoder = self.build_t2u_decoder()

        final_proj = TiedProjection(unit_embed.weight)

        return UnitYModel(
            s2t_model,
            t2u_encoder,
            t2u_decoder_frontend,
            t2u_decoder,
            final_proj,
            self.config.unit_pad_idx,
        )

    def build_t2u_decoder_frontend(self, embed: Embedding) -> TransformerFrontend:
        """Build a T2U Transformer decoder front-end."""
        return self.s2t_model_builder.build_decoder_frontend(embed)

    def build_t2u_encoder(self) -> TransformerEncoder:
        """Build a T2U Transformer encoder."""
        num_layers = self.config.num_t2u_encoder_layers

        layers = [self.build_t2u_encoder_layer() for _ in range(num_layers)]

        return StandardTransformerEncoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_t2u_decoder(self) -> TransformerDecoder:
        """Build a T2U Transformer decoder."""
        num_layers = self.config.num_t2u_decoder_layers

        layers = [self.build_t2u_decoder_layer() for _ in range(num_layers)]

        return StandardTransformerDecoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_t2u_encoder_layer(self) -> TransformerEncoderLayer:
        """Build a T2U Transformer encoder layer."""
        self_attn = self.build_attention(self.config.num_t2u_encoder_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.config.s2t_model_config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_t2u_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a T2U Transformer decoder layer."""
        self_attn = self.build_attention(self.config.num_t2u_decoder_attn_heads)

        encoder_decoder_attn = self.build_attention(
            self.config.num_t2u_decoder_attn_heads
        )

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn,
            ffn,
            dropout_p=self.config.s2t_model_config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        return self.s2t_model_builder.build_attention(num_heads)

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return self.s2t_model_builder.build_ffn()


def create_unity_model(
    config: UnitYConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> UnitYModel:
    """Create a UnitY model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    w2v2_encoder_builder = Wav2Vec2EncoderBuilder(
        config.s2t_model_config.w2v2_encoder_config, device, dtype
    )

    s2t_model_builder = UnitYS2TBuilder(
        config.s2t_model_config, w2v2_encoder_builder, device, dtype
    )

    return UnitYBuilder(config, s2t_model_builder, device, dtype).build_model()
