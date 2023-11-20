# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

from fairseq2.data import VocabularyInfo
from fairseq2.models.transformer import (
    TransformerDecoderModel,
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    init_final_projection,
)
from fairseq2.models.utils import ArchitectureRegistry
from fairseq2.nn.embedding import StandardEmbedding
from fairseq2.nn.normalization import LayerNorm, RMSNorm
from fairseq2.nn.position_encoder import RotaryEncoder
from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer import (
    CausalAttentionMaskFactory,
    FeedForwardNetwork,
    GLUFeedForwardNetwork,
    LocalAttentionStateFactory,
    MultiheadAttention,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.typing import DataType, Device


@dataclass
class MistralConfig:
    """Holds the configuration of a Mistral model."""

    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The expected maximum sequence length."""

    vocab_info: VocabularyInfo
    """The vocabulary information."""

    attn_window_len: int
    """The local attention window length."""

    num_layers: int
    """The number of Transformer decoder layers."""

    num_attn_heads: int
    """The number of attention heads in Transformer decoder layers."""

    num_key_value_heads: int
    """The number of key/value heads for Grouped Query Attention."""

    ffn_inner_dim: int
    """The dimensionality of inner projection layers in Transformer feed-forward
    networks."""

    dropout_p: float
    """The dropout probability in Transformer layers."""


mistral_archs = ArchitectureRegistry[MistralConfig]("mistral")

mistral_arch = mistral_archs.decorator


@mistral_arch("7b")
def _7b() -> MistralConfig:
    return MistralConfig(
        model_dim=4096,
        max_seq_len=8192,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        attn_window_len=4096,
        num_layers=32,
        num_attn_heads=32,
        num_key_value_heads=8,
        ffn_inner_dim=14336,
        dropout_p=0.1,
    )


class MistralBuilder:
    """Builds modules of a Mistral model as described in
    :cite:t:`https://doi.org/10.48550/arXiv.2310.06825`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: MistralConfig
    pos_encoder: Optional[RotaryEncoder]
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: MistralConfig,
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

        self.pos_encoder = None

        self.device, self.dtype = device, dtype

    def build_model(self) -> TransformerDecoderModel:
        """Build a model."""
        frontend = self.build_frontend()

        decoder = self.build_decoder()

        final_proj = Linear(
            self.config.model_dim,
            self.config.vocab_info.size,
            bias=False,
            init_fn=init_final_projection,
            device=self.device,
            dtype=self.dtype,
        )

        return TransformerDecoderModel(
            frontend, decoder, final_proj, self.config.vocab_info
        )

    def build_frontend(self) -> TransformerFrontend:
        """Build a Transformer decoder front-end."""
        embed = StandardEmbedding(
            num_embeddings=self.config.vocab_info.size,
            embedding_dim=self.config.model_dim,
            device=self.device,
            dtype=self.dtype,
        )

        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder=None,
            no_scale=True,  # Mistral does not use embedding scaling.
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        num_layers = self.config.num_layers

        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        self_attn_mask_factory = CausalAttentionMaskFactory(
            attn_window_len=self.config.attn_window_len
        )

        return StandardTransformerDecoder(
            layers,
            self_attn_mask_factory=self_attn_mask_factory,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.build_layer_norm,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        self_attn = self.build_attention(
            self.config.num_attn_heads, self.config.num_key_value_heads
        )

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn=None,
            ffn=ffn,
            dropout_p=self.config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.build_layer_norm,
            device=self.device,
            dtype=self.dtype,
        )

    def build_attention(
        self, num_heads: int, num_key_value_heads: int
    ) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        sdpa = create_default_sdpa(attn_dropout_p=self.config.dropout_p)

        if self.pos_encoder is None:
            self.pos_encoder = RotaryEncoder(
                self.config.model_dim // num_heads,
                self.config.max_seq_len,
                device=self.device,
            )

        state_factory = LocalAttentionStateFactory(self.config.attn_window_len)

        return StandardMultiheadAttention(
            self.config.model_dim,
            num_heads,
            num_key_value_heads=num_key_value_heads,
            sdpa=sdpa,
            pos_encoder=self.pos_encoder,
            bias=False,
            state_factory=state_factory,
            device=self.device,
            dtype=self.dtype,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return GLUFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            bias=False,
            inner_dim_scale=1.0,
            device=self.device,
            dtype=self.dtype,
        )

    def build_layer_norm(
        self,
        model_dim: int,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> LayerNorm:
        """Build a Layer Normalization module."""
        return RMSNorm(model_dim, bias=False, device=device, dtype=dtype)


def create_mistral_model(
    config: MistralConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> TransformerDecoderModel:
    """Create a Mistral model.

    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return MistralBuilder(config, device=device, dtype=dtype).build_model()
