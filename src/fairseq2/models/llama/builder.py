# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

from fairseq2.models.transformer import (
    FinalProjection,
    TransformerDecoderModel,
    TransformerEmbeddingFrontend,
    TransformerFrontend,
)
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.nn.embedding import Embedding
from fairseq2.nn.normalization import LayerNorm, RMSNorm
from fairseq2.nn.position_encoder import RotaryEncoder
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    SwiGLUFeedForwardNetwork,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.typing import DataType, Device


@dataclass
class LLaMAConfig:
    """Holds the configuration of an LLaMA model."""

    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The expected maximum sequence length."""

    vocabulary_size: int
    """The size of the vocabulary, one should define their own vocab size"""

    num_layers: int
    """The number of Transformer decoder layers."""

    num_attn_heads: int
    """The number of attention heads in Transformer decoder layers."""

    ffn_inner_dim: int
    """The dimensionality of inner projection layers in Transformer feed-forward
    networks."""

    ffn_round_to_multiple: int
    """The number that SwiGLU hidden dim is round to, should be a large power of 2"""

    dropout_p: float
    """The dropout probability in Transformer layers."""


llama_archs = ArchitectureRegistry[LLaMAConfig]("llama")


llama_arch = llama_archs.marker


@llama_arch("llama_7b")
def _llama_7b() -> LLaMAConfig:
    """7B model config"""
    return LLaMAConfig(
        model_dim=4096,
        max_seq_len=2048,
        vocabulary_size=32000,
        num_layers=32,
        num_attn_heads=32,
        ffn_inner_dim=4096 * 4,
        ffn_round_to_multiple=256,
        dropout_p=0.1,
    )


@llama_arch("llama_13b")
def _llama_13b() -> LLaMAConfig:
    """13B model config"""
    return LLaMAConfig(
        model_dim=5120,
        max_seq_len=2048,
        vocabulary_size=32000,
        num_layers=40,
        num_attn_heads=40,
        ffn_inner_dim=5120 * 4,
        ffn_round_to_multiple=256,
        dropout_p=0.1,
    )


@llama_arch("llama_33b")
def _llama_33b() -> LLaMAConfig:
    """33B model config"""
    return LLaMAConfig(
        model_dim=6656,
        max_seq_len=2048,
        vocabulary_size=32000,
        num_layers=60,
        num_attn_heads=52,
        ffn_inner_dim=6656 * 4,
        ffn_round_to_multiple=256,
        dropout_p=0.1,
    )


@llama_arch("llama_65b")
def _llama_65b() -> LLaMAConfig:
    """65B model config"""
    return LLaMAConfig(
        model_dim=8192,
        max_seq_len=2048,
        vocabulary_size=32000,
        num_layers=80,
        num_attn_heads=64,
        ffn_inner_dim=8192 * 4,
        ffn_round_to_multiple=256,
        dropout_p=0.1,
    )


class LLaMABuilder:
    """Builds modules of an LLaMA model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2302.13971`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: LLaMAConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: LLaMAConfig,
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

    def build_model(self) -> TransformerDecoderModel:
        """Build a LLaMA model."""
        frontend = self.build_frontend()

        decoder = self.build_decoder()

        final_proj = FinalProjection(
            self.config.model_dim,
            self.config.vocabulary_size,
            device=self.device,
            dtype=self.dtype,
        )

        return TransformerDecoderModel(
            frontend,
            decoder,
            final_proj,
            target_pad_idx=None,
        )

    def build_frontend(self) -> TransformerFrontend:
        """Build a LLaMA Transformer decoder front-end."""
        embed = Embedding(
            num_embeddings=self.config.vocabulary_size,
            embedding_dim=self.config.model_dim,
            device=self.device,
            dtype=self.dtype,
        )

        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder=None,
            no_scale=True,  # LLaMA does not use embedding scaling
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a LLaMA Transformer decoder."""
        num_layers = self.config.num_layers

        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        rms_norm_fn = self.build_layer_norm

        return StandardTransformerDecoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_fn=rms_norm_fn,
            device=self.device,
            dtype=self.dtype,
        )

    @staticmethod
    def build_layer_norm(
        model_dim: int,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> LayerNorm:
        """Constructs an RMSNorm layer."""
        return RMSNorm(
            model_dim,
            bias=False,  # LLaMA does not use bias parameters by default
            device=device,
            dtype=dtype,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a LLaMA Transformer decoder layer."""
        self_attn = self.build_attention(self.config.num_attn_heads)

        rms_norm_fn = self.build_layer_norm

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn=None,
            ffn=ffn,
            dropout_p=self.config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_fn=rms_norm_fn,
            device=self.device,
            dtype=self.dtype,
        )

    def build_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a LLaMA Transformer multi-head attention layer."""
        sdpa = create_default_sdpa(attn_dropout_p=self.config.dropout_p)

        pos_encoder = RotaryEncoder(
            self.config.model_dim // num_heads,
            self.config.max_seq_len,
            device=self.device,
            dtype=self.dtype,
        )

        return StandardMultiheadAttention(
            self.config.model_dim,
            num_heads,
            sdpa=sdpa,
            pos_encoder=pos_encoder,
            bias=False,  # LLaMA does not use bias parameters by default
            device=self.device,
            dtype=self.dtype,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a LLaMA Transformer feed-forward network."""
        return SwiGLUFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            self.config.ffn_round_to_multiple,
            device=self.device,
            dtype=self.dtype,
        )


def create_llama_model(
    config: LLaMAConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> TransformerDecoderModel:
    """Create a LLaMA model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return LLaMABuilder(config, device, dtype).build_model()
