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
from fairseq2.models.transformer import (
    TransformerDecoderModel,
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    init_final_projection,
)
from fairseq2.nn import LayerNorm, Linear, RMSNorm, RotaryEncoder, StandardEmbedding
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

MISTRAL_FAMILY: Final = "mistral"


@dataclass(kw_only=True)
class MistralConfig:
    """Holds the configuration of a Mistral model.

    The default values correspond to the base architecture as described in
    :cite:t:`https://doi.org/10.48550/arXiv.2310.06825`.
    """

    model_dim: int = 4096
    """The dimensionality of the model."""

    max_seq_len: int = 8192
    """The maximum sequence length."""

    vocab_info: VocabularyInfo = field(
        default_factory=lambda: VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        )
    )
    """The vocabulary information."""

    attn_window_len: int = 4096
    """The local attention window length."""

    num_layers: int = 32
    """The number of decoder layers."""

    num_attn_heads: int = 32
    """The number of attention heads in decoder layers."""

    num_key_value_heads: int = 8
    """The number of key/value heads for Grouped Query Attention."""

    ffn_inner_dim: int = 14336
    """The dimensionality of inner projection layers in feed-forward networks."""

    dropout_p: float = 0.1
    """The dropout probability on outputs of Transformer layers."""


mistral_archs = ConfigRegistry[MistralConfig]()

mistral_arch = mistral_archs.decorator


class MistralBuilder:
    """Builds modules of a Mistral model as described in
    :cite:t:`https://doi.org/10.48550/arXiv.2310.06825`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    _config: MistralConfig
    _device: Device | None
    _dtype: DataType | None
    _pos_encoder: RotaryEncoder | None

    def __init__(
        self,
        config: MistralConfig,
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

        self._pos_encoder = None

    def build_model(self) -> TransformerDecoderModel:
        """Build a model."""
        decoder_frontend = self.build_decoder_frontend()

        decoder = self.build_decoder()

        final_proj = Linear(
            self._config.model_dim,
            self._config.vocab_info.size,
            bias=False,
            init_fn=init_final_projection,
            device=self._device,
            dtype=self._dtype,
        )

        model = TransformerDecoderModel(
            decoder_frontend,
            decoder,
            final_proj,
            self._config.max_seq_len,
            self._config.vocab_info,
        )

        model.set_family(MISTRAL_FAMILY)

        return model

    def build_decoder_frontend(self) -> TransformerFrontend:
        """Build a Transformer decoder front-end."""
        embed = StandardEmbedding(
            num_embeddings=self._config.vocab_info.size,
            embedding_dim=self._config.model_dim,
            device=self._device,
            dtype=self._dtype,
        )

        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder=None,
            no_scale=True,  # Mistral does not use embedding scaling.
            dropout_p=self._config.dropout_p,
            device=self._device,
            dtype=self._dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        num_layers = self._config.num_layers

        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        self_attn_mask_factory = CausalAttentionMaskFactory(
            attn_window_len=self._config.attn_window_len
        )

        return StandardTransformerDecoder(
            layers,
            self_attn_mask_factory=self_attn_mask_factory,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.build_layer_norm,
            device=self._device,
            dtype=self._dtype,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        self_attn = self.build_attention(
            self._config.num_attn_heads, self._config.num_key_value_heads
        )

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn=None,
            ffn=ffn,
            dropout_p=self._config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.build_layer_norm,
            device=self._device,
            dtype=self._dtype,
        )

    def build_attention(
        self, num_heads: int, num_key_value_heads: int
    ) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        sdpa = create_default_sdpa(attn_dropout_p=self._config.dropout_p)

        if self._pos_encoder is None:
            self._pos_encoder = RotaryEncoder(
                self._config.model_dim // num_heads,
                self._config.max_seq_len,
                device=self._device,
            )

        state_factory = LocalAttentionStateFactory(self._config.attn_window_len)

        return StandardMultiheadAttention(
            self._config.model_dim,
            num_heads,
            num_key_value_heads=num_key_value_heads,
            sdpa=sdpa,
            pos_encoder=self._pos_encoder,
            bias=False,
            state_factory=state_factory,
            device=self._device,
            dtype=self._dtype,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return GLUFeedForwardNetwork(
            self._config.model_dim,
            self._config.ffn_inner_dim,
            bias=False,
            inner_dim_scale=1.0,
            device=self._device,
            dtype=self._dtype,
        )

    def build_layer_norm(
        self,
        model_dim: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> LayerNorm:
        """Build a Layer Normalization module."""
        return RMSNorm(model_dim, bias=False, device=device, dtype=dtype)


def create_mistral_model(
    config: MistralConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> TransformerDecoderModel:
    """Create a Mistral model."""
    return MistralBuilder(config, device=device, dtype=dtype).build_model()


model_factories.register(
    MISTRAL_FAMILY, create_mistral_model, MistralConfig, mistral_archs
)
