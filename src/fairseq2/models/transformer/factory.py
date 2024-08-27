# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from fairseq2.config_registry import ConfigRegistry
from fairseq2.data import VocabularyInfo
from fairseq2.models.factory import model_factories
from fairseq2.models.transformer.frontend import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
)
from fairseq2.models.transformer.model import TransformerModel
from fairseq2.nn import (
    Embedding,
    SinusoidalPositionEncoder,
    StandardEmbedding,
    TiedProjection,
    init_scaled_embedding,
)
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

TRANSFORMER_FAMILY: Final = "transformer"


@dataclass(kw_only=True)
class TransformerConfig:
    """Holds the configuration of a Transformer model.

    The default values correspond to the base architecture as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.
    """

    model_dim: int = 512
    """The dimensionality of the model."""

    max_seq_len: int = 1024
    """The maximum sequence length."""

    vocab_info: VocabularyInfo = field(
        default_factory=lambda: VocabularyInfo(
            size=32768, unk_idx=None, bos_idx=None, eos_idx=1, pad_idx=0
        )
    )
    """The vocabulary information."""

    num_encoder_layers: int = 6
    """The number of encoder layers."""

    num_decoder_layers: int = 6
    """The number of decoder layers."""

    num_encoder_attn_heads: int = 8
    """The number of attention heads in encoder layers."""

    num_decoder_attn_heads: int = 8
    """The number of attention heads in decoder layers."""

    ffn_inner_dim: int = 2048
    """The dimensionality of inner projection layers in feed-forward networks."""

    norm_order: TransformerNormOrder = TransformerNormOrder.POST
    """The Layer Normalization order."""

    dropout_p: float = 0.1
    """The dropout probability on outputs of Transformer layers."""


transformer_archs = ConfigRegistry[TransformerConfig]()

transformer_arch = transformer_archs.decorator


class TransformerBuilder:
    """Builds modules of a Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    _config: TransformerConfig
    _device: Device | None
    _dtype: DataType | None

    def __init__(
        self,
        config: TransformerConfig,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param config:
            The configuration.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self._config = config

        self._device, self._dtype = device, dtype

    def build_model(self) -> TransformerModel:
        """Build a model."""
        embed = self.build_embedding()

        frontend = self.build_frontend(embed)

        encoder = self.build_encoder()
        decoder = self.build_decoder()

        final_proj = TiedProjection(embed.weight, bias=None)

        model = TransformerModel(
            frontend,
            encoder,
            frontend,
            decoder,
            final_proj,
            self._config.max_seq_len,
            self._config.vocab_info,
        )

        model.set_family(TRANSFORMER_FAMILY)

        return model

    def build_embedding(self) -> StandardEmbedding:
        """Build an embedding table."""
        return StandardEmbedding(
            num_embeddings=self._config.vocab_info.size,
            embedding_dim=self._config.model_dim,
            pad_idx=self._config.vocab_info.pad_idx,
            init_fn=init_scaled_embedding,
            device=self._device,
            dtype=self._dtype,
        )

    def build_frontend(self, embed: Embedding) -> TransformerFrontend:
        """Build a Transformer encoder/decoder front-end."""
        pos_encoder = SinusoidalPositionEncoder(
            self._config.model_dim,
            self._config.max_seq_len,
            _legacy_pad_idx=1,
            device=self._device,
        )

        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder,
            dropout_p=self._config.dropout_p,
            device=self._device,
            dtype=self._dtype,
        )

    def build_encoder(self) -> TransformerEncoder:
        """Build a Transformer encoder."""
        num_layers = self._config.num_encoder_layers

        layers = [self.build_encoder_layer() for _ in range(num_layers)]

        return StandardTransformerEncoder(
            layers,
            norm_order=self._config.norm_order,
            device=self._device,
            dtype=self._dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        num_layers = self._config.num_decoder_layers

        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        return StandardTransformerDecoder(
            layers,
            norm_order=self._config.norm_order,
            device=self._device,
            dtype=self._dtype,
        )

    def build_encoder_layer(self) -> TransformerEncoderLayer:
        """Build a Transformer encoder layer."""
        self_attn = self.build_attention(self._config.num_encoder_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self._config.dropout_p,
            norm_order=self._config.norm_order,
            device=self._device,
            dtype=self._dtype,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        self_attn = self.build_attention(self._config.num_decoder_attn_heads)

        encoder_decoder_attn = self.build_attention(self._config.num_decoder_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn,
            ffn,
            dropout_p=self._config.dropout_p,
            norm_order=self._config.norm_order,
            device=self._device,
            dtype=self._dtype,
        )

    def build_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        sdpa = create_default_sdpa(attn_dropout_p=self._config.dropout_p)

        return StandardMultiheadAttention(
            self._config.model_dim,
            num_heads,
            sdpa=sdpa,
            device=self._device,
            dtype=self._dtype,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self._config.model_dim,
            self._config.ffn_inner_dim,
            bias=True,
            norm_order=self._config.norm_order,
            device=self._device,
            dtype=self._dtype,
        )


def create_transformer_model(
    config: TransformerConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> TransformerModel:
    """Create a Transformer model."""
    return TransformerBuilder(config, device=device, dtype=dtype).build_model()


model_factories.register(
    TRANSFORMER_FAMILY, create_transformer_model, TransformerConfig, transformer_archs
)
