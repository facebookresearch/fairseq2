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
from fairseq2.nn.lora import LoRAConfig
from fairseq2.nn.normalization import LayerNorm, RMSNorm
from fairseq2.nn.position_encoder import RotaryEncoder
from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    GLUFeedForwardNetwork,
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
class LLaMAConfig:
    """Holds the configuration of a LLaMA model."""

    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The expected maximum sequence length."""

    vocab_info: VocabularyInfo
    """The vocabulary information."""

    num_layers: int
    """The number of Transformer decoder layers."""

    num_attn_heads: int
    """The number of attention heads in Transformer decoder layers."""

    num_key_value_heads: int
    """The number of key/value heads for Grouped Query Attention."""

    ffn_inner_dim: int
    """The dimensionality of inner projection layers in Transformer feed-forward
    networks."""

    ffn_inner_dim_to_multiple: int
    """The dimensionality of inner projection layers in Transformer feed-forward
    networks is rounded up to the nearest multiple of this value."""

    dropout_p: float
    """The dropout probability in Transformer layers."""


llama_archs = ArchitectureRegistry[LLaMAConfig]("llama")

llama_arch = llama_archs.decorator


@llama_arch("7b")
def _7b() -> LLaMAConfig:
    return LLaMAConfig(
        model_dim=4096,
        max_seq_len=2048,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=32,
        num_attn_heads=32,
        num_key_value_heads=32,
        ffn_inner_dim=4096 * 4,
        ffn_inner_dim_to_multiple=256,
        dropout_p=0.1,
    )


@llama_arch("13b")
def _13b() -> LLaMAConfig:
    return LLaMAConfig(
        model_dim=5120,
        max_seq_len=2048,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=40,
        num_attn_heads=40,
        num_key_value_heads=40,
        ffn_inner_dim=5120 * 4,
        ffn_inner_dim_to_multiple=256,
        dropout_p=0.1,
    )


@llama_arch("33b")
def _33b() -> LLaMAConfig:
    return LLaMAConfig(
        model_dim=6656,
        max_seq_len=2048,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=60,
        num_attn_heads=52,
        num_key_value_heads=52,
        ffn_inner_dim=6656 * 4,
        ffn_inner_dim_to_multiple=256,
        dropout_p=0.1,
    )


@llama_arch("65b")
def _65b() -> LLaMAConfig:
    return LLaMAConfig(
        model_dim=8192,
        max_seq_len=2048,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=80,
        num_attn_heads=64,
        num_key_value_heads=64,
        ffn_inner_dim=8192 * 4,
        ffn_inner_dim_to_multiple=256,
        dropout_p=0.1,
    )


@llama_arch("llama2_7b")
def _llama2_7b() -> LLaMAConfig:
    return LLaMAConfig(
        model_dim=4096,
        max_seq_len=4096,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=32,
        num_attn_heads=32,
        num_key_value_heads=32,
        ffn_inner_dim=4096 * 4,
        ffn_inner_dim_to_multiple=256,
        dropout_p=0.1,
    )


@llama_arch("llama2_13b")
def _llama2_13b() -> LLaMAConfig:
    return LLaMAConfig(
        model_dim=5120,
        max_seq_len=4096,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=40,
        num_attn_heads=40,
        num_key_value_heads=40,
        ffn_inner_dim=5120 * 4,
        ffn_inner_dim_to_multiple=256,
        dropout_p=0.1,
    )


@llama_arch("llama2_70b")
def _llama2_70b() -> LLaMAConfig:
    return LLaMAConfig(
        model_dim=8192,
        max_seq_len=4096,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=80,
        num_attn_heads=64,
        num_key_value_heads=8,
        ffn_inner_dim=int(8192 * 4 * 1.3),  # See A.2.1 in LLaMA 2
        ffn_inner_dim_to_multiple=4096,
        dropout_p=0.1,
    )


class LLaMABuilder:
    """Builds modules of a LLaMA model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2302.13971` and
    :cite:t:`https://doi.org/10.48550/arXiv.2307.09288`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: LLaMAConfig
    pos_encoder: Optional[RotaryEncoder]
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: LLaMAConfig,
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
            no_scale=True,  # LLaMA does not use embedding scaling.
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        num_layers = self.config.num_layers

        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        return StandardTransformerDecoder(
            layers,
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

        return StandardMultiheadAttention(
            self.config.model_dim,
            num_heads,
            num_key_value_heads=num_key_value_heads,
            sdpa=sdpa,
            pos_encoder=self.pos_encoder,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return GLUFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            bias=False,
            inner_dim_to_multiple=self.config.ffn_inner_dim_to_multiple,
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


def create_llama_model(
    config: LLaMAConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> TransformerDecoderModel:
    """Create a LLaMA model.

    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return LLaMABuilder(config, device=device, dtype=dtype).build_model()


def get_llama_lora_config() -> LoRAConfig:
    return LoRAConfig(
        r=8,
        alpha=16.0,
        dropout_p=0.05,
        keys=[".*decoder.layers.*.self_attn.*(q_proj|v_proj)$"],
    )
