# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import torch
from torch import Tensor

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
from fairseq2.nn.lora import LoRAConfig
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

LLAMA_FAMILY: Final = "llama"


@dataclass(kw_only=True)
class LLaMAConfig:
    """Holds the configuration of a LLaMA model.

    The default values correspond to the base architecture as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2302.13971`.
    """

    model_dim: int = 4096
    """The dimensionality of the model."""

    max_seq_len: int = 2048
    """The maximum sequence length."""

    vocab_info: VocabularyInfo = field(
        default_factory=lambda: VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        )
    )
    """The vocabulary information."""

    num_layers: int = 32
    """The number of decoder layers."""

    num_attn_heads: int = 32
    """The number of attention heads in decoder layers."""

    num_key_value_heads: int = 32
    """The number of key/value heads for Grouped Query Attention."""

    ffn_inner_dim: int = 4096 * 4
    """The dimensionality of inner projection layers in feed-forward networks."""

    ffn_inner_dim_scale: float = 2 / 3
    """The scale factor for the dimensionality of inner projection layers in
    feed forward networks."""

    ffn_inner_dim_to_multiple: int = 256
    """The dimensionality of inner projection layers in feed-forward networks is
    rounded up to the nearest multiple of this value."""

    rope_theta: float = 10_000.0
    """The coefficient of the long-term decay of the Rotary position encoder."""

    use_scaled_rope: bool = False
    """If ``True``, scales Rotary encoding frequencies to LLaMA 3.1 context length."""

    dropout_p: float = 0.1
    """The dropout probability on outputs of Transformer layers."""


llama_archs = ConfigRegistry[LLaMAConfig]()

llama_arch = llama_archs.decorator


class LLaMABuilder:
    """Builds modules of a LLaMA model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2302.13971` and
    :cite:t:`https://doi.org/10.48550/arXiv.2307.09288`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    _config: LLaMAConfig
    _device: Device | None
    _dtype: DataType | None
    _pos_encoder: RotaryEncoder | None

    def __init__(
        self,
        config: LLaMAConfig,
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

        model.set_family(LLAMA_FAMILY)

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
            no_scale=True,  # LLaMA does not use embedding scaling.
            dropout_p=self._config.dropout_p,
            device=self._device,
            dtype=self._dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        num_layers = self._config.num_layers

        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        return StandardTransformerDecoder(
            layers,
            dropout_p=self._config.dropout_p,
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
            if self._config.use_scaled_rope:
                freqs_init_fn = self._init_scaled_freqs
            else:
                freqs_init_fn = None

            self._pos_encoder = RotaryEncoder(
                self._config.model_dim // num_heads,
                self._config.max_seq_len,
                theta=self._config.rope_theta,
                freqs_init_fn=freqs_init_fn,
                device=self._device,
            )

        return StandardMultiheadAttention(
            self._config.model_dim,
            num_heads,
            num_key_value_heads=num_key_value_heads,
            sdpa=sdpa,
            pos_encoder=self._pos_encoder,
            bias=False,
            device=self._device,
            dtype=self._dtype,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return GLUFeedForwardNetwork(
            self._config.model_dim,
            self._config.ffn_inner_dim,
            bias=False,
            inner_dim_scale=self._config.ffn_inner_dim_scale,
            inner_dim_to_multiple=self._config.ffn_inner_dim_to_multiple,
            inner_dropout_p=self._config.dropout_p,
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

    @staticmethod
    def _init_scaled_freqs(pos_encoder: RotaryEncoder) -> Tensor:
        device = pos_encoder.freqs.device

        # (E / 2)
        indices = torch.arange(
            0, pos_encoder.encoding_dim, step=2, device=device, dtype=torch.float32
        )

        freqs = 1.0 / (pos_encoder.theta ** (indices / pos_encoder.encoding_dim))

        if device.type == "meta":
            return freqs  # type: ignore[no-any-return]

        old_context_len = 8192  # The context length of LLaMA 3.

        scale_factor = 8.0

        l_freq_factor = 1
        h_freq_factor = 5

        l_freq_wavelen = old_context_len / l_freq_factor
        h_freq_wavelen = old_context_len / h_freq_factor

        new_freqs = []

        for freq in freqs.tolist():
            wavelen = 2 * math.pi / freq

            if wavelen < h_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > l_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                smooth = (old_context_len / wavelen - l_freq_factor) / (h_freq_factor - l_freq_factor)  # fmt: skip
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

        return torch.tensor(new_freqs, dtype=freqs.dtype, device=device)


def create_llama_model(
    config: LLaMAConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> TransformerDecoderModel:
    """Create a LLaMA model."""
    return LLaMABuilder(config, device=device, dtype=dtype).build_model()


model_factories.register(LLAMA_FAMILY, create_llama_model, LLaMAConfig, llama_archs)


def get_llama_lora_config() -> LoRAConfig:
    return LoRAConfig(
        r=8,
        alpha=16.0,
        dropout_p=0.05,
        keys=[".*decoder.layers.*.self_attn.*(q_proj|v_proj)$"],
    )
