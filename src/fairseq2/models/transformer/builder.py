# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

from fairseq2.models.transformer.arch import (
    ScoreProjection,
    Transformer,
    TransformerTokenFrontend,
)
from fairseq2.nn.embedding import Embedding
from fairseq2.nn.positional_embedding import (
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from fairseq2.nn.projection import Projection, TiedProjection
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
class TransformerConfig:
    """The default arguments correspond to the *base* Transformer model as
    described in Table 3 of :cite:t:`DBLP:journals/corr/VaswaniSPUJGKP17`."""

    src_num_tokens: int
    """The number of source tokens, e.g. vocabulary size."""

    tgt_num_tokens: int
    """The number of target tokens, e.g. vocabulary size."""

    src_padding_token_idx: Optional[int]
    """If not ``None``, entries at ``src_padding_token_idx`` in source sequences
    won't contribute to the gradient."""

    tgt_padding_token_idx: Optional[int]
    """If not ``None``, entries at ``tgt_padding_token_idx`` in target sequences
    won't contribute to the gradient."""

    max_src_len: int = 1024
    """The expected maximum source sequence length."""

    max_tgt_len: int = 1024
    """The expected maximum target sequence length."""

    model_dim: int = 512
    """The dimensionality of the model (i.e. inputs and outputs)."""

    share_embed: bool = True
    """If ``True``, the encoder and decoder embeddings will share the same
    weight."""

    share_dec_input_output: bool = True
    """If ``True``, the decoder embedding and pre-softmax output projection will
    share the same weight."""

    num_enc_layers: int = 6
    """The number of encoder layers."""

    num_dec_layers: int = 6
    """The number of decoder layers."""

    num_enc_attn_heads: int = 8
    """The number of attention heads in encoder layers."""

    num_dec_attn_heads: int = 8
    """The number of attention heads in decoder layers."""

    ffn_inner_dim: int = 2048
    """The dimensionality of inner layers in feed-forward networks."""

    ffn_inner_dropout_p: float = 0.0
    """The dropout probability on outputs of inner layers in feed-forward
    networks."""

    pre_layer_norm: bool = False
    """If ``True``, Layer Normalization will be applied at the beginning of each
    layer as described in :cite:t:`DBLP:journals/corr/abs-2002-04745`."""

    dropout_p: float = 0.1
    """The dropout probability on outputs of embedding layers, attention layers,
    and feed-forward networks."""

    attn_dropout_p: float = 0.0
    """The dropout probability on attention weights."""

    legacy_pos_embed: bool = False
    """If ``True``, sinusoidal positional embeddings will be initialized in a
    way that is compatible with the original fairseq."""


class TransformerBuilder:
    """Builds a model that follows the Transformer architecture as described in
    :cite:t:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    To tweak the model architecture, you can subclass this builder and override
    the corresponding methods.
    """

    cfg: TransformerConfig
    norm_order: TransformerNormOrder
    enc_embed: Optional[Embedding]
    dec_embed: Optional[Embedding]
    enc_pos_embed: Optional[PositionalEmbedding]
    dec_pos_embed: Optional[PositionalEmbedding]

    def __init__(self, cfg: TransformerConfig) -> None:
        """
        :param cfg:
            The configuration to use.
        """
        self.cfg = cfg

        if cfg.share_embed:
            if cfg.src_num_tokens != cfg.tgt_num_tokens:
                raise ValueError(
                    f"`src_num_tokens` ({cfg.src_num_tokens}) and `tgt_num_tokens` ({cfg.tgt_num_tokens}) must match when `share_embed` is `True`."
                )

            if cfg.max_src_len != cfg.max_tgt_len:
                raise ValueError(
                    f"`max_src_len` ({cfg.max_src_len}) and `max_tgt_len` ({cfg.max_tgt_len}) must match when `share_embed` is `True`."
                )

            if cfg.src_padding_token_idx != cfg.tgt_padding_token_idx:
                raise ValueError(
                    f"`src_padding_token_idx` ({cfg.src_padding_token_idx}) and `tgt_padding_token_idx` ({cfg.tgt_padding_token_idx}) must match when `share_embed` is `True`."
                )

        if cfg.pre_layer_norm:
            self.norm_order = TransformerNormOrder.PRE
        else:
            self.norm_order = TransformerNormOrder.POST

    def build(self, device=None, dtype=None) -> Transformer:
        """Build a model."""
        self.device, self.dtype = device, dtype

        enc_frontend = self._build_enc_frontend()
        dec_frontend = self._build_dec_frontend()

        enc = self._build_encoder()
        dec = self._build_decoder()

        score_proj = self._build_score_projection()

        model = Transformer(enc_frontend, enc, dec_frontend, dec, score_proj)

        self.enc_embed = None
        self.dec_embed = None

        self.enc_pos_embed = None
        self.dec_pos_embed = None

        return model

    def _build_enc_frontend(self) -> TransformerTokenFrontend:
        """Build the encoder front-end."""
        self.enc_embed = self._build_enc_embedding()

        self.enc_pos_embed = self._build_enc_positional_embedding()

        return TransformerTokenFrontend(
            embed=self.enc_embed,
            pos_embed=self.enc_pos_embed,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def _build_dec_frontend(self) -> TransformerTokenFrontend:
        """Build the decoder front-end."""
        self.dec_embed = self._build_dec_embedding()

        self.dec_pos_embed = self._build_dec_positional_embedding()

        return TransformerTokenFrontend(
            embed=self.dec_embed,
            pos_embed=self.dec_pos_embed,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def _build_enc_embedding(self) -> Embedding:
        """Build the encoder embedding."""
        return self._build_embedding(
            self.cfg.src_num_tokens, self.cfg.src_padding_token_idx
        )

    def _build_dec_embedding(self) -> Embedding:
        """Build the decoder embedding."""
        if self.cfg.share_embed:
            assert self.enc_embed is not None

            return self.enc_embed

        return self._build_embedding(
            self.cfg.tgt_num_tokens, self.cfg.tgt_padding_token_idx
        )

    def _build_embedding(self, num_embed: int, padding_idx: Optional[int]) -> Embedding:
        """Build an embedding."""
        return Embedding(
            num_embed=num_embed,
            embedding_dim=self.cfg.model_dim,
            padding_idx=padding_idx,
            scaled=True,
            device=self.device,
            dtype=self.dtype,
        )

    def _build_enc_positional_embedding(self) -> PositionalEmbedding:
        """Build the encoder positional embedding."""
        return self._build_positional_embedding(
            self.cfg.max_src_len, self.cfg.src_padding_token_idx
        )

    def _build_dec_positional_embedding(self) -> PositionalEmbedding:
        """Build the decoder positional embedding."""
        if self.cfg.share_embed:
            assert self.enc_pos_embed is not None

            return self.enc_pos_embed

        return self._build_positional_embedding(
            self.cfg.max_tgt_len, self.cfg.tgt_padding_token_idx
        )

    def _build_positional_embedding(
        self, max_seq_len: int, padding_token_idx: Optional[int]
    ) -> PositionalEmbedding:
        """Build a positional embedding."""
        if self.cfg.legacy_pos_embed:
            padding_token_idx = padding_token_idx or 0
        else:
            padding_token_idx = None

        return SinusoidalPositionalEmbedding(
            max_seq_len=max_seq_len,
            embedding_dim=self.cfg.model_dim,
            legacy_padding_idx=padding_token_idx,
            device=self.device,
            dtype=self.dtype,
        )

    def _build_encoder(self) -> TransformerEncoder:
        """Build the encoder."""
        layers = [self._build_encoder_layer(i) for i in range(self.cfg.num_enc_layers)]

        return StandardTransformerEncoder(
            layers, norm_order=self.norm_order, device=self.device, dtype=self.dtype
        )

    def _build_decoder(self) -> TransformerDecoder:
        """Build the decoder."""
        layers = [self._build_decoder_layer(i) for i in range(self.cfg.num_dec_layers)]

        return StandardTransformerDecoder(
            layers, norm_order=self.norm_order, device=self.device, dtype=self.dtype
        )

    def _build_encoder_layer(self, idx: int) -> TransformerEncoderLayer:
        """Build an encoder layer.

        :param idx:
            The index of the layer in the encoder stack.
        """
        self_attn = self._build_encoder_attn()

        ffn = self._build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn=self_attn,
            ffn=ffn,
            dropout_p=self.cfg.dropout_p,
            norm_order=self.norm_order,
            device=self.device,
            dtype=self.dtype,
        )

    def _build_decoder_layer(self, idx: int) -> TransformerDecoderLayer:
        """Build a decoder layer.

        :param idx:
            The index of the layer in the decoder stack.
        """
        self_attn = self._build_decoder_attn()

        enc_dec_attn = self._build_encoder_decoder_attn()

        ffn = self._build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn=self_attn,
            enc_dec_attn=enc_dec_attn,
            ffn=ffn,
            dropout_p=self.cfg.dropout_p,
            norm_order=self.norm_order,
            device=self.device,
            dtype=self.dtype,
        )

    def _build_encoder_attn(self) -> MultiheadAttention:
        """Build a self attention layer to be used in the encoder stack."""
        return self._build_attn(self.cfg.num_enc_attn_heads)

    def _build_decoder_attn(self) -> MultiheadAttention:
        """Build a self attention layer to be used in the decoder stack."""
        return self._build_attn(self.cfg.num_dec_attn_heads)

    def _build_encoder_decoder_attn(self) -> MultiheadAttention:
        """Build an encoder-decoder attention layer."""
        return self._build_attn(self.cfg.num_dec_attn_heads)

    def _build_attn(self, num_heads: int) -> MultiheadAttention:
        """Build a multi-head attention layer."""
        return StandardMultiheadAttention(
            num_heads=num_heads,
            model_dim=self.cfg.model_dim,
            attn_dropout_p=self.cfg.attn_dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def _build_ffn(self) -> FeedForwardNetwork:
        """Build a feed-forward network."""
        return StandardFeedForwardNetwork(
            model_dim=self.cfg.model_dim,
            inner_dim=self.cfg.ffn_inner_dim,
            inner_dropout_p=self.cfg.ffn_inner_dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def _build_score_projection(self) -> Projection:
        """Build the pre-softmax score projection."""
        if self.cfg.share_dec_input_output:
            assert self.dec_embed is not None

            return TiedProjection(self.dec_embed.weight)

        return ScoreProjection(
            num_embed=self.cfg.tgt_num_tokens,
            embedding_dim=self.cfg.model_dim,
            device=self.device,
            dtype=self.dtype,
        )


def build_transformer(cfg: TransformerConfig, device=None, dtype=None) -> Transformer:
    """Build a model that follows the Transformer architecture as described in
    :cite:t:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    :param cfg:
        The configuration to use.
    """
    return TransformerBuilder(cfg).build(device, dtype)


def transformer_iwslt_de_en() -> TransformerConfig:
    return TransformerConfig(
        src_num_tokens=8848,
        tgt_num_tokens=6632,
        src_padding_token_idx=1,
        tgt_padding_token_idx=1,
        share_embed=False,
        share_dec_input_output=False,
        num_enc_attn_heads=4,
        num_dec_attn_heads=4,
        ffn_inner_dim=1024,
    )
