# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal, Optional

from fairseq2.data import VocabularyInfo
from fairseq2.models.transformer import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    TransformerModel,
)
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.nn.embedding import Embedding
from fairseq2.nn.position_encoder import (
    LearnedPositionEncoder,
    SinusoidalPositionEncoder,
)
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
    create_default_sdpa,
)
from fairseq2.typing import DataType, Device


@dataclass
class mBartConfig:
    """Holds the configuration of an mBart model."""

    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The expected maximum sequence length."""

    vocabulary_size: int
    """The size of the vocabulary."""

    pad_idx: Optional[int]
    """The index of the pad symbol in the vocabulary."""

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

    # Position Encoder
    pos_encoder_type: Literal["sinusoidal", "learned"]
    """The type of position encoder."""

    layer_norm_embed: bool
    """Adds a layernorm to the embedding in the Transformer encoder."""

    dropout_p: float
    """The dropout probability in Transformer layers."""

    def update_vocabulary(self, info: VocabularyInfo) -> None:
        """Update vocabulary configuration from ``info``."""
        self.vocabulary_size, self.pad_idx = info.size, info.pad_idx


mbart_archs = ArchitectureRegistry[mBartConfig]("mbart")


mbart_arch = mbart_archs.marker


@mbart_arch("base")
def _base() -> mBartConfig:
    return mBartConfig(
        model_dim=1024,
        max_seq_len=1026,
        vocabulary_size=65539,
        pad_idx=0,
        num_encoder_layers=12,
        num_decoder_layers=12,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=4096,
        pos_encoder_type="learned",
        layer_norm_embed=True,
        dropout_p=0.1,
    )


class mBartBuilder:
    """Builds modules of an mBart model as described in
    :cite:t:`https://arxiv.org/abs/2001.08210`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: mBartConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: mBartConfig,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config
        self.device = device
        self.dtype = dtype

    def build_model(self) -> TransformerModel:
        """Build a model."""
        embed = self.build_embedding()

        frontend = self.build_frontend(embed)

        encoder = self.build_encoder()
        decoder = self.build_decoder()

        final_proj = TiedProjection(embed.weight)

        return TransformerModel(
            frontend, encoder, frontend, decoder, final_proj, self.config.pad_idx
        )

    def build_embedding(self) -> Embedding:
        """Build an embedding table."""
        return Embedding(
            num_embeddings=self.config.vocabulary_size,
            embedding_dim=self.config.model_dim,
            pad_idx=self.config.pad_idx,
            scaled=True,
            device=self.device,
            dtype=self.dtype,
        )

    def build_frontend(self, embed: Embedding) -> TransformerFrontend:
        """Build a Transformer encoder/decoder front-end."""
        if self.config.pos_encoder_type == "sinusoidal":
            pos_encoder = SinusoidalPositionEncoder(
                self.config.model_dim,
                self.config.max_seq_len,
                _legacy_pad_idx=self.config.pad_idx,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            pos_encoder = LearnedPositionEncoder(
                self.config.model_dim,
                self.config.max_seq_len,
                device=self.device,
                dtype=self.dtype,
            )

        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder,
            layer_norm=self.config.layer_norm_embed,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_encoder(self) -> TransformerEncoder:
        """Build a Transformer encoder."""
        num_layers = self.config.num_encoder_layers

        layers = [self.build_encoder_layer() for _ in range(num_layers)]

        return StandardTransformerEncoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
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
        self_attn = self.build_attention(self.config.num_encoder_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.config.dropout_p,
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
        sdpa = create_default_sdpa(attn_dropout_p=self.config.dropout_p)

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


def create_mbart_model(
    config: mBartConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> TransformerModel:
    """Create an mBart model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return mBartBuilder(config, device, dtype).build_model()
