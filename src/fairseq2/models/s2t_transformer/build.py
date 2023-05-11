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
from fairseq2.models.conformer import ConformerBlock, ConformerConvolution
from fairseq2.models.s2t_transformer.feature_extractor import Conv1dFbankSubsampler
from fairseq2.models.s2t_transformer.frontend import S2TTransformerFrontend
from fairseq2.models.transformer import (
    FinalProjection,
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    TransformerModel,
)
from fairseq2.nn.embedding import Embedding
from fairseq2.nn.position_encoder import PositionEncoder, SinusoidalPositionEncoder
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
    """The dimensionality of the model."""

    use_conformer: bool = False
    """If ``True``, uses Conformer blocks instead of Transformer encoder layers."""

    num_encoder_layers: int = 12
    """The number of Transformer encoder layers."""

    num_decoder_layers: int = 6
    """The number of Transformer decoder layers."""

    num_encoder_attn_heads: int = 8
    """The number of attention heads in Transformer encoder layers."""

    num_decoder_attn_heads: int = 8
    """The number of attention heads in Transformer decoder layers."""

    ffn_inner_dim: int = 512 * 4
    """The dimensionality of inner projection layers in Transformer feed-forward
    networks."""

    dropout_p: float = 0.15
    """The dropout probability in Transformer layers."""

    depthwise_conv_kernel_size: int = 31
    """The kernel size of depthwise convolutions in Conformer blocks."""


_CONFIGS: Final = {
    "tiny": lambda: S2TTransformerConfig(
        model_dim=256,
        num_encoder_layers=6,
        num_decoder_layers=3,
        num_encoder_attn_heads=4,
        num_decoder_attn_heads=4,
        ffn_inner_dim=256 * 4,
        dropout_p=0.3,
    ),
    "small": lambda: S2TTransformerConfig(
        model_dim=256,
        num_encoder_layers=12,
        num_decoder_layers=6,
        num_encoder_attn_heads=4,
        num_decoder_attn_heads=4,
        ffn_inner_dim=256 * 8,
        dropout_p=0.1,
    ),
    "medium": lambda: S2TTransformerConfig(
        model_dim=512,
        num_encoder_layers=12,
        num_decoder_layers=6,
        num_encoder_attn_heads=8,
        num_decoder_attn_heads=8,
        ffn_inner_dim=512 * 4,
        dropout_p=0.15,
    ),
    "large": lambda: S2TTransformerConfig(
        model_dim=1024,
        num_encoder_layers=12,
        num_decoder_layers=6,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=1024 * 4,
        dropout_p=0.2,
    ),
    "conformer": lambda: S2TTransformerConfig(
        max_seq_len=6000,
        model_dim=256,
        use_conformer=True,
        num_encoder_layers=12,
        num_decoder_layers=6,
        num_encoder_attn_heads=4,
        num_decoder_attn_heads=8,
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
    dtype: Optional[torch.dtype] = None,
) -> TransformerModel:
    """Create an S2T Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`.

    :param cfg:
        The configuration to use.
    :param vocab_info:
        The vocabulary information to use.
    :param device:
        The device on which to initialize the model.
    :param dtype:
        The data type of the model parameters and buffers.
    """
    return S2TTransformerBuilder(cfg, vocab_info, device, dtype).build_model()


class S2TTransformerBuilder:
    """Builds modules of an S2T Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`.

    To tweak the model architecture, you can derive from this class and override
    the corresponding methods.
    """

    cfg: S2TTransformerConfig
    vocab_info: VocabularyInfo
    encoder_norm_order: TransformerNormOrder
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]

    def __init__(
        self,
        cfg: S2TTransformerConfig,
        vocab_info: VocabularyInfo,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param cfg:
            The configuration to use.
        :param vocab_info:
            The vocabulary information to use.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.cfg = cfg
        self.vocab_info = vocab_info

        if self.cfg.use_conformer:
            self.encoder_norm_order = TransformerNormOrder.POST
        else:
            self.encoder_norm_order = TransformerNormOrder.PRE

        self.device = device
        self.dtype = dtype

    def build_model(self) -> TransformerModel:
        """Build a model."""
        encoder_frontend = self.build_encoder_frontend()
        decoder_frontend = self.build_decoder_frontend()

        encoder = self.build_encoder()
        decoder = self.build_decoder()

        final_proj = FinalProjection(
            model_dim=self.cfg.model_dim,
            target_vocab_size=self.vocab_info.size,
            device=self.device,
            dtype=self.dtype,
        )

        return TransformerModel(
            encoder_frontend,
            encoder,
            decoder_frontend,
            decoder,
            final_proj,
            self.vocab_info.pad_idx,
        )

    def build_encoder_frontend(self) -> TransformerFrontend:
        """Build a Transformer encoder front-end."""
        feat_extractor = Conv1dFbankSubsampler(
            num_channels=self.cfg.num_fbank_channels,
            inner_dim=1024,
            out_dim=self.cfg.model_dim,
            kernel_sizes=[5, 5],
            device=self.device,
            dtype=self.dtype,
        )

        pos_encoder = self.build_position_encoder()

        return S2TTransformerFrontend(
            self.cfg.model_dim,
            feat_extractor,
            pos_encoder,
            proj=self.cfg.use_conformer,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder_frontend(self) -> TransformerFrontend:
        """Build a Transformer decoder front-end."""
        embed = Embedding(
            num_embeddings=self.vocab_info.size,
            embedding_dim=self.cfg.model_dim,
            pad_idx=self.vocab_info.pad_idx,
            scaled=True,
            device=self.device,
            dtype=self.dtype,
        )

        pos_encoder = self.build_position_encoder()

        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_position_encoder(self) -> PositionEncoder:
        """Build a position encoder."""
        return SinusoidalPositionEncoder(
            self.cfg.model_dim,
            self.cfg.max_seq_len,
            _legacy_pad_idx=self.vocab_info.pad_idx,
            device=self.device,
            dtype=self.dtype,
        )

    def build_encoder(self) -> TransformerEncoder:
        """Build a Transformer encoder."""
        layers = [
            self.build_encoder_layer() for _ in range(self.cfg.num_encoder_layers)
        ]

        return StandardTransformerEncoder(
            layers,
            norm_order=self.encoder_norm_order,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        layers = [
            self.build_decoder_layer() for _ in range(self.cfg.num_decoder_layers)
        ]

        return StandardTransformerDecoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_encoder_layer(self) -> TransformerEncoderLayer:
        """Build a Transformer encoder layer."""
        if self.cfg.use_conformer:
            return self.build_conformer_block()

        self_attn = self.build_attention(self.cfg.num_encoder_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.cfg.dropout_p,
            norm_order=self.encoder_norm_order,
            device=self.device,
            dtype=self.dtype,
        )

    def build_conformer_block(self) -> TransformerEncoderLayer:
        """Build a Conformer block."""
        ffn1 = self.build_ffn(use_swish=True)

        self_attn = self.build_attention(self.cfg.num_encoder_attn_heads)

        conv = ConformerConvolution(
            self.cfg.model_dim,
            self.cfg.depthwise_conv_kernel_size,
            device=self.device,
            dtype=self.dtype,
        )

        ffn2 = self.build_ffn(use_swish=True)

        return ConformerBlock(
            ffn1,
            self_attn,
            conv,
            ffn2,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        self_attn = self.build_attention(self.cfg.num_decoder_attn_heads)

        encoder_decoder_attn = self.build_attention(self.cfg.num_decoder_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn,
            ffn,
            dropout_p=self.cfg.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        return StandardMultiheadAttention(
            num_heads,
            self.cfg.model_dim,
            attn_dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_ffn(self, use_swish: bool = False) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self.cfg.model_dim,
            self.cfg.ffn_inner_dim,
            inner_activation=SiLU() if use_swish else None,
            inner_dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )
