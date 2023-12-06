# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

from torch.nn import SiLU

from fairseq2.data import VocabularyInfo
from fairseq2.models.conformer import ConformerBlock, ConformerConvolution
from fairseq2.models.s2t_transformer.feature_extractor import Conv1dFbankSubsampler
from fairseq2.models.s2t_transformer.frontend import S2TTransformerFrontend
from fairseq2.models.transformer import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    TransformerModel,
    init_final_projection,
)
from fairseq2.models.utils import ArchitectureRegistry
from fairseq2.nn.embedding import StandardEmbedding, init_scaled_embedding
from fairseq2.nn.position_encoder import PositionEncoder, SinusoidalPositionEncoder
from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    RelativePositionalEncoding,
    RelativePositionSDPA,
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
    create_default_sdpa,
)
from fairseq2.typing import DataType, Device


@dataclass
class S2TTransformerConfig:
    """Holds the configuration of an S2T Transformer model."""

    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The expected maximum source sequence length after feature extraction."""

    num_fbank_channels: int
    """The number of source log-mel filterbank channels."""

    target_vocab_info: VocabularyInfo
    """The target vocabulary information."""

    use_relative_pos: bool
    """If ``True``, uses relative positional encodings for source sequences."""

    use_conformer: bool
    """If ``True``, uses Conformer blocks instead of Transformer encoder layers."""

    num_encoder_layers: int
    """The number of Transformer encoder layers."""

    num_decoder_layers: int
    """The number of Transformer decoder layers."""

    num_encoder_attn_heads: int
    """The number of attention heads in Transformer encoder layers."""

    num_decoder_attn_heads: int
    """The number of attention heads in Transformer decoder layers."""

    ffn_inner_dim: int
    """The inner dimensionality of Transformer feed-forward networks."""

    dropout_p: float
    """The dropout probability in Transformer layers."""

    depthwise_conv_kernel_size: int
    """The kernel size of depthwise convolutions in Conformer blocks."""


s2t_transformer_archs = ArchitectureRegistry[S2TTransformerConfig]("s2t_transformer")

s2t_transformer_arch = s2t_transformer_archs.decorator


@s2t_transformer_arch("tiny")
def _tiny() -> S2TTransformerConfig:
    return S2TTransformerConfig(
        model_dim=256,
        max_seq_len=1024,
        num_fbank_channels=80,
        target_vocab_info=VocabularyInfo(
            size=10000, unk_idx=0, bos_idx=0, eos_idx=0, pad_idx=1
        ),
        use_relative_pos=False,
        use_conformer=False,
        num_encoder_layers=6,
        num_decoder_layers=3,
        num_encoder_attn_heads=4,
        num_decoder_attn_heads=4,
        ffn_inner_dim=256 * 4,
        dropout_p=0.3,
        depthwise_conv_kernel_size=0,
    )


@s2t_transformer_arch("small")
def _small() -> S2TTransformerConfig:
    return S2TTransformerConfig(
        model_dim=256,
        max_seq_len=1024,
        num_fbank_channels=80,
        target_vocab_info=VocabularyInfo(
            size=10000, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1
        ),
        use_relative_pos=False,
        use_conformer=False,
        num_encoder_layers=12,
        num_decoder_layers=6,
        num_encoder_attn_heads=4,
        num_decoder_attn_heads=4,
        ffn_inner_dim=256 * 8,
        dropout_p=0.1,
        depthwise_conv_kernel_size=0,
    )


@s2t_transformer_arch("medium")
def _medium() -> S2TTransformerConfig:
    return S2TTransformerConfig(
        model_dim=512,
        max_seq_len=1024,
        num_fbank_channels=80,
        target_vocab_info=VocabularyInfo(
            size=10000, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1
        ),
        use_relative_pos=False,
        use_conformer=False,
        num_encoder_layers=12,
        num_decoder_layers=6,
        num_encoder_attn_heads=8,
        num_decoder_attn_heads=8,
        ffn_inner_dim=512 * 4,
        dropout_p=0.15,
        depthwise_conv_kernel_size=0,
    )


@s2t_transformer_arch("large")
def _large() -> S2TTransformerConfig:
    return S2TTransformerConfig(
        model_dim=1024,
        max_seq_len=1024,
        num_fbank_channels=80,
        target_vocab_info=VocabularyInfo(
            size=10000, unk_idx=0, bos_idx=0, eos_idx=0, pad_idx=1
        ),
        use_relative_pos=False,
        use_conformer=False,
        num_encoder_layers=12,
        num_decoder_layers=6,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=1024 * 4,
        dropout_p=0.2,
        depthwise_conv_kernel_size=0,
    )


@s2t_transformer_arch("conformer_medium")
def _conformer_medium() -> S2TTransformerConfig:
    return S2TTransformerConfig(
        model_dim=256,
        max_seq_len=6000,
        num_fbank_channels=80,
        target_vocab_info=VocabularyInfo(
            size=181, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1
        ),
        use_relative_pos=False,
        use_conformer=True,
        num_encoder_layers=12,
        num_decoder_layers=6,
        num_encoder_attn_heads=4,
        num_decoder_attn_heads=8,
        ffn_inner_dim=512 * 4,
        dropout_p=0.1,
        depthwise_conv_kernel_size=31,
    )


class S2TTransformerBuilder:
    """Builds modules of an S2T Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: S2TTransformerConfig
    rel_pos_encoding: Optional[RelativePositionalEncoding]
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: S2TTransformerConfig,
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
        self.config = config

        self.rel_pos_encoding = None

        self.device, self.dtype = device, dtype

    def build_model(self) -> TransformerModel:
        """Build a model."""
        encoder_frontend = self.build_encoder_frontend()
        encoder = self.build_encoder()

        decoder_frontend = self.build_decoder_frontend()
        decoder = self.build_decoder()

        final_proj = Linear(
            self.config.model_dim,
            self.config.target_vocab_info.size,
            bias=False,
            init_fn=init_final_projection,
            device=self.device,
            dtype=self.dtype,
        )

        return TransformerModel(
            encoder_frontend,
            encoder,
            decoder_frontend,
            decoder,
            final_proj,
            self.config.target_vocab_info,
        )

    def build_encoder_frontend(self) -> TransformerFrontend:
        """Build a Transformer encoder front-end."""
        feat_extractor = Conv1dFbankSubsampler(
            num_channels=self.config.num_fbank_channels,
            inner_dim=1024,
            feature_dim=self.config.model_dim,
            kernel_sizes=[5, 5],
            device=self.device,
            dtype=self.dtype,
        )

        pos_encoder = self.build_source_position_encoder()

        return S2TTransformerFrontend(
            self.config.model_dim,
            feat_extractor,
            pos_encoder,
            proj=self.config.use_conformer,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder_frontend(self) -> TransformerFrontend:
        """Build a Transformer decoder front-end."""
        embed = StandardEmbedding(
            num_embeddings=self.config.target_vocab_info.size,
            embedding_dim=self.config.model_dim,
            pad_idx=self.config.target_vocab_info.pad_idx,
            init_fn=init_scaled_embedding,
            device=self.device,
            dtype=self.dtype,
        )

        pos_encoder = self.build_target_position_encoder()

        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_source_position_encoder(self) -> Optional[PositionEncoder]:
        """Build a position encoder for source sequences."""
        if self.config.use_relative_pos:
            return None

        return SinusoidalPositionEncoder(
            self.config.model_dim, self.config.max_seq_len, device=self.device
        )

    def build_target_position_encoder(self) -> PositionEncoder:
        """Build a position encoder for target sequences."""
        return SinusoidalPositionEncoder(
            self.config.model_dim,
            self.config.max_seq_len,
            _legacy_pad_idx=1,
            device=self.device,
        )

    def build_encoder(self) -> TransformerEncoder:
        """Build a Transformer encoder."""
        num_layers = self.config.num_encoder_layers

        layers = [self.build_encoder_layer() for _ in range(num_layers)]

        if self.config.use_conformer:
            encoder_norm_order = TransformerNormOrder.POST
        else:
            encoder_norm_order = TransformerNormOrder.PRE

        return StandardTransformerEncoder(
            layers, norm_order=encoder_norm_order, device=self.device, dtype=self.dtype
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

    def build_encoder_layer(self) -> TransformerEncoderLayer:
        """Build a Transformer encoder layer."""
        if self.config.use_conformer:
            return self.build_conformer_block()

        self_attn = self.build_encoder_attention()

        ffn = self.build_ffn()

        if self.config.use_conformer:
            encoder_norm_order = TransformerNormOrder.POST
        else:
            encoder_norm_order = TransformerNormOrder.PRE

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.config.dropout_p,
            norm_order=encoder_norm_order,
            device=self.device,
            dtype=self.dtype,
        )

    def build_conformer_block(self) -> TransformerEncoderLayer:
        """Build a Conformer block."""
        ffn1 = self.build_ffn(use_swish=True)

        self_attn = self.build_encoder_attention()

        conv = ConformerConvolution(
            self.config.model_dim,
            self.config.depthwise_conv_kernel_size,
            device=self.device,
            dtype=self.dtype,
        )

        ffn2 = self.build_ffn(use_swish=True)

        return ConformerBlock(
            ffn1,
            self_attn,
            conv,
            ffn2,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        self_attn = self.build_decoder_attention()

        encoder_decoder_attn = self.build_decoder_attention()

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

    def build_encoder_attention(self) -> MultiheadAttention:
        """Build a Transformer encoder multi-head attention layer."""
        sdpa = create_default_sdpa(attn_dropout_p=self.config.dropout_p)

        if self.config.use_relative_pos:
            if self.rel_pos_encoding is None:
                self.rel_pos_encoding = RelativePositionalEncoding(
                    self.config.model_dim,
                    self.config.max_seq_len,
                    device=self.device,
                    dtype=self.dtype,
                )

            sdpa = RelativePositionSDPA(
                self.config.model_dim,
                self.config.num_encoder_attn_heads,
                self.rel_pos_encoding,
                inner_sdpa=sdpa,
                device=self.device,
                dtype=self.dtype,
            )

        return StandardMultiheadAttention(
            self.config.model_dim,
            self.config.num_encoder_attn_heads,
            sdpa=sdpa,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder_attention(self) -> MultiheadAttention:
        """Build a Transformer decoder multi-head attention layer."""
        sdpa = create_default_sdpa(attn_dropout_p=self.config.dropout_p)

        return StandardMultiheadAttention(
            self.config.model_dim,
            self.config.num_decoder_attn_heads,
            sdpa=sdpa,
            device=self.device,
            dtype=self.dtype,
        )

    def build_ffn(self, use_swish: bool = False) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            bias=True,
            inner_activation=SiLU() if use_swish else None,
            inner_dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )


def create_s2t_transformer_model(
    config: S2TTransformerConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> TransformerModel:
    """Create an S2T Transformer model.

    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return S2TTransformerBuilder(config, device=device, dtype=dtype).build_model()
