# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Final, Optional

from fairseq2.data import VocabularyInfo
from fairseq2.models.architecture_registry import ModelArchitectureRegistry
from fairseq2.models.transformer import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    TransformerModel,
)
from fairseq2.nn.embedding import Embedding, StandardEmbedding, init_scaled_embedding
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
    create_default_sdpa,
)
from fairseq2.typing import DataType, Device

NLLB_FAMILY: Final = "nllb"


@dataclass
class NllbConfig:
    """Holds the configuration of an NLLB model.

    The default values correspond to the dense 1B architecture as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2207.04672`.
    """

    model_dim: int = 1024
    """The dimensionality of the model."""

    max_seq_len: int = 1024
    """The maximum allowed sequence length."""

    vocab_info: VocabularyInfo = VocabularyInfo(
        size=256206, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=0
    )
    """The vocabulary information."""

    num_encoder_layers: int = 24
    """The number of encoder layers."""

    num_decoder_layers: int = 24
    """The number of decoder layers."""

    num_encoder_attn_heads: int = 16
    """The number of attention heads in encoder layers."""

    num_decoder_attn_heads: int = 16
    """The number of attention heads in decoder layers."""

    ffn_inner_dim: int = 1024 * 8
    """The dimensionality of inner projection layers in feed-forward networks."""

    dropout_p: float = 0.1
    """The dropout probability on outputs of Transformer layers."""


nllb_archs = ModelArchitectureRegistry[NllbConfig]()

nllb_arch = nllb_archs.decorator


@nllb_arch("dense_600m")
def _dense_600m() -> NllbConfig:
    config = _dense_1b()

    config.num_encoder_layers = 12
    config.num_decoder_layers = 12
    config.ffn_inner_dim = 1024 * 4

    return config


@nllb_arch("dense_1b")
def _dense_1b() -> NllbConfig:
    return NllbConfig()


@nllb_arch("dense_3b")
def _dense_3b() -> NllbConfig:
    config = _dense_1b()

    config.model_dim = 2048

    return config


class NllbBuilder:
    """Builds modules of an NLLB model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2207.04672`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    _config: NllbConfig
    _device: Optional[Device]
    _dtype: Optional[DataType]

    def __init__(
        self,
        config: NllbConfig,
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
        self._config = config

        self._device, self._dtype = device, dtype

    def build_model(self) -> TransformerModel:
        """Build a model."""
        embed = self.build_embedding()

        frontend = self.build_frontend(embed)

        encoder = self.build_encoder()
        decoder = self.build_decoder()

        final_proj = TiedProjection(embed.weight, bias=None)

        return TransformerModel(
            frontend,
            encoder,
            frontend,
            decoder,
            final_proj,
            self._config.max_seq_len,
            self._config.vocab_info,
        )

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
            norm_order=TransformerNormOrder.PRE,
            device=self._device,
            dtype=self._dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        num_layers = self._config.num_decoder_layers

        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        return StandardTransformerDecoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
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
            norm_order=TransformerNormOrder.PRE,
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
            norm_order=TransformerNormOrder.PRE,
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
            norm_order=TransformerNormOrder.PRE,
            device=self._device,
            dtype=self._dtype,
        )


def create_nllb_model(
    config: NllbConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> TransformerModel:
    """Create an NLLB model.

    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    model = NllbBuilder(config, device=device, dtype=dtype).build_model()

    return model.set_family(NLLB_FAMILY)
