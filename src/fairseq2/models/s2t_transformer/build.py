# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import AbstractSet, Final, Optional

import torch
from torch.nn import SiLU

from fairseq2.data.text import VocabularyInfo
from fairseq2.models.conformer import ConformerConvolution, ConformerEncoderLayer
from fairseq2.models.s2t_transformer.model import (
    S2TTransformerModel,
    TransformerFbankFrontend,
)
from fairseq2.models.s2t_transformer.subsampler import Conv1dFbankSubsampler
from fairseq2.models.transformer import ScoreProjection, TransformerTokenFrontend
from fairseq2.nn.embedding import Embedding
from fairseq2.nn.positional_embedding import (
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
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
)


@dataclass
class S2TTransformerConfig:
    """Holds the configuration of an S2T Transformer model.

    The default values correspond to the *medium* architecture described in
    Table 3 of :cite:t:`https://doi.org/10.48550/arxiv.2010.05171`.
    """

    num_fbank_channels: int = 80
    """The number of source log-mel filterbank channels."""

    max_seq_len: int = 1024
    """The expected maximum sequence length."""

    model_dim: int = 512
    """The dimensionality of the model (i.e. inputs and outputs)."""

    use_conformer: bool = False
    """If ``True``, uses Conformer blocks instead of Transformer encoder layers."""

    num_enc_layers: int = 12
    """The number of encoder layers."""

    num_dec_layers: int = 6
    """The number of decoder layers."""

    num_enc_attn_heads: int = 8
    """The number of attention heads in encoder layers."""

    num_dec_attn_heads: int = 8
    """The number of attention heads in decoder layers."""

    ffn_inner_dim: int = 512 * 4
    """The dimensionality of inner projection layers in feed-forward networks."""

    dropout_p: float = 0.15
    """The dropout probability on outputs of embedding dictionaries, attention
    layers, and feed-forward networks."""

    depthwise_conv_kernel_size: int = 31
    """The kernel size of depthwise convolutions in Conformer blocks."""

    dtype: torch.dtype = torch.float32
    """The data type of model parameters and buffers."""


_CONFIGS: Final = {
    "tiny": lambda: S2TTransformerConfig(
        model_dim=256,
        num_enc_layers=6,
        num_dec_layers=3,
        num_enc_attn_heads=4,
        num_dec_attn_heads=4,
        ffn_inner_dim=256 * 4,
        dropout_p=0.3,
    ),
    "small": lambda: S2TTransformerConfig(
        model_dim=256,
        num_enc_layers=12,
        num_dec_layers=6,
        num_enc_attn_heads=4,
        num_dec_attn_heads=4,
        ffn_inner_dim=256 * 8,
        dropout_p=0.1,
    ),
    "medium": lambda: S2TTransformerConfig(
        model_dim=512,
        num_enc_layers=12,
        num_dec_layers=6,
        num_enc_attn_heads=8,
        num_dec_attn_heads=8,
        ffn_inner_dim=512 * 4,
        dropout_p=0.15,
    ),
    "large": lambda: S2TTransformerConfig(
        model_dim=1024,
        num_enc_layers=12,
        num_dec_layers=6,
        num_enc_attn_heads=16,
        num_dec_attn_heads=16,
        ffn_inner_dim=1024 * 4,
        dropout_p=0.2,
    ),
    "conformer": lambda: S2TTransformerConfig(
        max_seq_len=6000,
        model_dim=256,
        use_conformer=True,
        num_enc_layers=12,
        num_dec_layers=6,
        num_enc_attn_heads=4,
        num_dec_attn_heads=8,
        ffn_inner_dim=512 * 4,
        dropout_p=0.1,
    ),
}


def get_s2t_transformer_archs() -> AbstractSet[str]:
    """Return the names of supported S2T Transformer architectures."""
    return _CONFIGS.keys()


def get_s2t_transformer_config(arch_name: str) -> S2TTransformerConfig:
    """Return the configuration of the specified S2T Transformer architecture.

    :param arch_name:
        The name of the architecture.
    """
    try:
        return _CONFIGS[arch_name]()
    except KeyError:
        raise ValueError(
            f"`arch_name` must be a known S2T Transformer architecture, but is '{arch_name}' instead."
        )


def create_s2t_transformer_model(
    cfg: S2TTransformerConfig,
    vocab_info: VocabularyInfo,
    device: Optional[torch.device] = None,
) -> S2TTransformerModel:
    """Create an S2T Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`.

    :param cfg:
        The configuration to use.
    :param vocab_info:
        The vocabulary information to use.
    :param device:
        The device on which to initialize the model.
    """
    return S2TTransformerBuilder(cfg, vocab_info, device).build_model()


class S2TTransformerBuilder:
    """Builds modules of an S2T Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`.

    To tweak the model architecture, you can derive from this class and override
    the corresponding methods.
    """

    cfg: S2TTransformerConfig
    vocab_info: VocabularyInfo
    device: Optional[torch.device]

    def __init__(
        self,
        cfg: S2TTransformerConfig,
        vocab_info: VocabularyInfo,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        :param cfg:
            The configuration to use.
        :param vocab_info:
            The vocabulary information to use.
        :param device:
            The device on which to initialize modules.
        """
        self.cfg = cfg
        self.vocab_info = vocab_info
        self.device = device

    def build_model(self) -> S2TTransformerModel:
        """Build a model."""
        enc_frontend = self.build_encoder_frontend()
        dec_frontend = self.build_decoder_frontend()

        enc = self.build_encoder()
        dec = self.build_decoder()

        score_proj = ScoreProjection(
            num_embed=self.vocab_info.size,
            model_dim=self.cfg.model_dim,
            device=self.device,
            dtype=self.cfg.dtype,
        )

        return S2TTransformerModel(enc_frontend, enc, dec_frontend, dec, score_proj)

    def build_encoder_frontend(self) -> TransformerFbankFrontend:
        """Build an encoder frontend."""
        subsampler = Conv1dFbankSubsampler(
            num_channels=self.cfg.num_fbank_channels,
            inner_dim=1024,
            embed_dim=self.cfg.model_dim,
            kernel_sizes=[5, 5],
            device=self.device,
            dtype=self.cfg.dtype,
        )

        pos_embed = self.build_positional_embedding()

        return TransformerFbankFrontend(
            subsampler,
            pos_embed,
            apply_projection=self.cfg.use_conformer,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_decoder_frontend(self) -> TransformerTokenFrontend:
        """Build a decoder frontend."""
        embed = Embedding(
            num_embed=self.vocab_info.size,
            embed_dim=self.cfg.model_dim,
            pad_idx=self.vocab_info.pad_idx,
            scaled=True,
            device=self.device,
            dtype=self.cfg.dtype,
        )

        pos_embed = self.build_positional_embedding()

        return TransformerTokenFrontend(
            embed,
            pos_embed,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_positional_embedding(self) -> PositionalEmbedding:
        """Build a positional embedding."""
        return SinusoidalPositionalEmbedding(
            max_seq_len=self.cfg.max_seq_len,
            embed_dim=self.cfg.model_dim,
            legacy_pad_token_idx=self.vocab_info.pad_idx,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_encoder(self) -> TransformerEncoder:
        """Build an encoder."""
        layers = [self.build_encoder_layer() for _ in range(self.cfg.num_enc_layers)]

        if not self.cfg.use_conformer:
            norm_order = TransformerNormOrder.PRE
        else:
            # We do not apply Layer Normalization to the output of the encoder
            # since Conformer blocks already apply it.
            norm_order = TransformerNormOrder.POST

        return StandardTransformerEncoder(
            layers, norm_order=norm_order, device=self.device, dtype=self.cfg.dtype
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a decoder."""
        layers = [self.build_decoder_layer() for _ in range(self.cfg.num_dec_layers)]

        return StandardTransformerDecoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_encoder_layer(self) -> TransformerEncoderLayer:
        """Build an encoder layer."""
        if self.cfg.use_conformer:
            return self.build_conformer_block()

        self_attn = self.build_attention(self.cfg.num_enc_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.cfg.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_conformer_block(self) -> TransformerEncoderLayer:
        """Build a Conformer block."""
        ffn1 = self.build_ffn(use_swish=True)

        self_attn = self.build_attention(self.cfg.num_enc_attn_heads)

        conv = ConformerConvolution(
            self.cfg.model_dim,
            self.cfg.depthwise_conv_kernel_size,
            device=self.device,
            dtype=self.cfg.dtype,
        )

        ffn2 = self.build_ffn(use_swish=True)

        return ConformerEncoderLayer(
            ffn1,
            self_attn,
            conv,
            ffn2,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a decoder layer."""
        self_attn = self.build_attention(self.cfg.num_dec_attn_heads)

        enc_dec_attn = self.build_attention(self.cfg.num_dec_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            enc_dec_attn,
            ffn,
            dropout_p=self.cfg.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a multi-head attention layer."""
        return StandardMultiheadAttention(
            num_heads,
            self.cfg.model_dim,
            attn_dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_ffn(self, use_swish: bool = False) -> FeedForwardNetwork:
        """Build a feed-forward network."""
        return StandardFeedForwardNetwork(
            self.cfg.model_dim,
            self.cfg.ffn_inner_dim,
            inner_activation=SiLU() if use_swish else None,
            inner_dropout_p=self.cfg.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.cfg.dtype,
        )
