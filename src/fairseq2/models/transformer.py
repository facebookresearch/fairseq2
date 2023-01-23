# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["Transformer", "TransformerBuilder"]

from typing import Any, Dict, Optional, final

from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Module

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
from fairseq2.typing import DataType, Device


@final
class Transformer(Module):
    """Represents a Transformer model as described in
    :cite:t:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    """

    model_dim: int
    """The dimensionality of the model (i.e. inputs and outputs)."""

    batch_first: bool
    """If ``True``, the first dimension of batched inputs and outputs represents
    the batch; otherwise, the sequence."""

    encoder: TransformerEncoder
    """The encoder."""

    decoder: TransformerDecoder
    """The decoder."""

    score_proj: Projection
    """The projection to apply to the outputs of the decoder."""

    def __init__(
        self,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        score_proj: Projection,
    ) -> None:
        """
        :param encoder:
            The encoder.
        :param decoder:
            The decoder.
        :param score_proj:
            The projection to apply to the outputs of the decoder.
        """
        if encoder.model_dim != decoder.model_dim:
            raise ValueError(
                f"`model_dim` of `encoder` ({encoder.model_dim}) does not match `model_dim` of `decoder` ({decoder.model_dim})."
            )

        if encoder.batch_first != decoder.batch_first:
            raise ValueError(
                f"`batch_first` of `encoder` ({encoder.batch_first}) does not match `batch_first` of `decoder` ({decoder.batch_first})."
            )

        super().__init__()

        self.model_dim = encoder.model_dim

        self.batch_first = encoder.batch_first

        self.encoder = encoder
        self.decoder = decoder

        self.score_proj = score_proj

    @finaloverride
    def forward(self, src_seq: Tensor, tgt_seq: Tensor) -> Tensor:
        """
        :param src_seq:
            The source sequences. *Shape:* :math:`(S)` when unbatched,
            :math:`(N,S)` when :attr:`batch_first` is ``True``, or :math:`(S,N)`
            when :attr:`batch_first` is ``False``, where :math:`N` is the batch
            size and :math:`S` is the source sequence length.
        :param tgt_seq:
            The target sequences. *Shape:* :math:`(T)` when unbatched,
            :math:`(N,T)` when :attr:`batch_first` is ``True``, or :math:`(T,N)`
            when :attr:`batch_first` is ``False``, where :math:`N` is the batch
            size and :math:`T` is the target sequence length.

        :returns:
            The output of :attr:`score_proj`. The produced scores should be
            forwarded to a softmax function to compute the next-token
            probabilities. *Shape:* :math:`(T,D)` when
            unbatched, :math:`(N,T,D)` when :attr:`batch_first` is ``True``, or
            :math:`(T,N,D)` when :attr:`batch_first` is ``False``, where
            :math:`N` is the batch size, :attr:`T` is the target sequence
            length, and :math:`D` is the size of the output embedding
            dictionary.
        """
        enc_out, enc_attn_padding_mask = self.encoder(src_seq)

        x = self.decoder(tgt_seq, enc_out, enc_attn_padding_mask)

        x = self.score_proj(x)

        return x  # type: ignore[no-any-return]


class TransformerBuilder:
    """Builds Transformer models as described in
    :cite:t:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    If you want to tweak the model architecture, derive from this class and
    override the corresponding methods.
    """

    num_tokens: int
    """The number of tokens, e.g. vocabulary size."""

    max_seq_len: int
    """The expected maximum sequence length."""

    padding_token_idx: Optional[int]
    """If not ``None``, entries at :attr:`padding_token_idx` won't contribute
    to the gradient."""

    model_dim: int
    """The dimensionality of the model (i.e. inputs and outputs)."""

    num_enc_layers: int
    """The number of encoder layers."""

    num_dec_layers: int
    """The number of decoder layers."""

    num_enc_attn_heads: int
    """The number of attention heads in encoder layers."""

    num_dec_attn_heads: int
    """The number of attention heads in decoder layers."""

    ffn_inner_dim: int
    """The dimensionality of inner layers in feed-forward networks."""

    dropout_p: float
    """The dropout probability on the outputs of attention layers, feed-forward
    networks, and input/output embeddings."""

    norm_order: TransformerNormOrder
    """The Layer Normalization order to use."""

    batch_first: bool
    """If ``True``, the first dimension of batched inputs and outputs represents
    the batch; otherwise, the sequence."""

    device: Optional[Device]
    """The device on which to build the model."""

    dtype: Optional[DataType]
    """The floating-point type of model parameters."""

    _fct_kwargs: Dict[str, Any]

    def __init__(
        self,
        num_tokens: int,
        max_seq_len: int = 4096,
        padding_token_idx: Optional[int] = None,
        model_dim: int = 512,
        num_enc_layers: int = 6,
        num_dec_layers: int = 6,
        num_enc_attn_heads: int = 8,
        num_dec_attn_heads: int = 8,
        ffn_inner_dim: int = 2048,
        dropout_p: float = 0.1,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        batch_first: bool = False,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        The default arguments correspond to the *base* Transformer model as
        described in Table 3 of :cite:t:`DBLP:journals/corr/VaswaniSPUJGKP17`.

        :param num_tokens:
            The number of tokens, e.g. vocabulary size.
        :param max_seq_length:
            The expected maximum sequence length.
        :param padding_token_idx:
            If not ``None``, entries at ``padding_token_idx`` won't contribute
            to the gradient.
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs).
        :param num_enc_layers:
            The number of encoder layers.
        :param num_dec_layers:
            The number of decoder layers.
        :param num_enc_attn_heads:
            The number of attention heads in encoder layers.
        :param num_dec_attn_heads:
            The number of attention heads in decoder layers.
        :param ffn_inner_dim:
            The dimensionality of inner layers in feed-forward networks.
        :param dropout_p:
            The dropout probability on the outputs of attention layers, feed-
            forward networks, and input/output embeddings.
        :param norm_order:
            The Layer Normalization order to use.
        :param batch_first:
            If ``True``, the first dimension of batched inputs and outputs
            represents the batch; otherwise, the sequence.
        :param device:
            The device on which to build the model.
        :param dtype:
            The floating-point type of model parameters.
        """
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.padding_token_idx = padding_token_idx
        self.model_dim = model_dim
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_enc_attn_heads = num_enc_attn_heads
        self.num_dec_attn_heads = num_dec_attn_heads
        self.ffn_inner_dim = ffn_inner_dim
        self.dropout_p = dropout_p
        self.norm_order = norm_order
        self.batch_first = batch_first
        self.device = device
        self.dtype = dtype

        # Holds common keyword arguments.
        self._fct_kwargs = {"device": device, "dtype": dtype}

    def build(self) -> Transformer:
        """Builds a :class:`Transformer` model."""
        embed = self.build_embedding()

        enc_pos_embed = self.build_positional_embedding()
        dec_pos_embed = self.build_positional_embedding()

        encoder = self.build_encoder(embed, enc_pos_embed)
        decoder = self.build_decoder(embed, dec_pos_embed)

        score_proj = TiedProjection(embed.weight)

        return Transformer(encoder, decoder, score_proj)

    def build_embedding(self) -> Embedding:
        """Builds an :class:`Embedding`."""
        return Embedding(
            num_embed=self.num_tokens,
            embedding_dim=self.model_dim,
            padding_idx=self.padding_token_idx,
            scaled=True,
            **self._fct_kwargs,
        )

    def build_positional_embedding(self) -> Optional[PositionalEmbedding]:
        """Builds an optional :class:`PositionalEmbedding`."""
        return SinusoidalPositionalEmbedding(
            max_seq_len=self.max_seq_len,
            embedding_dim=self.model_dim,
            padding_token_idx=self.padding_token_idx,
            batch_first=self.batch_first,
            **self._fct_kwargs,
        )

    def build_encoder(
        self, embed: Embedding, pos_embed: Optional[PositionalEmbedding]
    ) -> TransformerEncoder:
        """Builds a :class:`TransformerEncoder`.

        :param embed:
            The input :class:`Embedding`.
        :param pos_embed:
            The optional :class:`PositionalEmbedding` to use with ``embed``.
        """
        layers = [self.build_encoder_layer(i) for i in range(self.num_enc_layers)]

        return StandardTransformerEncoder(
            embed,
            pos_embed,
            layers,
            norm_order=self.norm_order,
            embed_dropout_p=self.dropout_p,
            **self._fct_kwargs,
        )

    def build_encoder_layer(self, idx: int) -> TransformerEncoderLayer:
        """Builds a :class:`TransformerEncoderLayer`.

        :param idx:
            The index of the layer in the stack.
        """
        self_attn = self.build_encoder_attn()

        ffn = self.build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.dropout_p,
            norm_order=self.norm_order,
            **self._fct_kwargs,
        )

    def build_encoder_attn(self) -> MultiheadAttention:
        """Builds a :class:`MultiheadAttention` for self attention in encoder
        layers."""
        return self.build_attn(self.num_enc_attn_heads)

    def build_decoder(
        self, embed: Embedding, pos_embed: Optional[PositionalEmbedding]
    ) -> TransformerDecoder:
        """Builds a :class:`TransformerDecoder`.

        :param embed:
            The output :class:`Embedding`.
        :param pos_embed:
            The optional :class:`PositionalEmbedding` to add to ``embed``.
        """
        layers = [self.build_decoder_layer(i) for i in range(self.num_dec_layers)]

        return StandardTransformerDecoder(
            embed,
            pos_embed,
            layers,
            norm_order=self.norm_order,
            embed_dropout_p=self.dropout_p,
            **self._fct_kwargs,
        )

    def build_decoder_layer(self, idx: int) -> TransformerDecoderLayer:
        """Builds a :class:`TransformerDecoderLayer`.

        :param idx:
            The index of the layer in the stack.
        """
        self_attn = self.build_decoder_attn()

        enc_dec_attn = self.build_encoder_decoder_attn()

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            enc_dec_attn,
            ffn,
            dropout_p=self.dropout_p,
            norm_order=self.norm_order,
            **self._fct_kwargs,
        )

    def build_decoder_attn(self) -> MultiheadAttention:
        """Builds a :class:`MultiheadAttention` for self attention in decoder
        layers."""
        return self.build_attn(self.num_dec_attn_heads)

    def build_encoder_decoder_attn(self) -> MultiheadAttention:
        """Builds a :class:`MultiheadAttention` for encoder-decoder attention in
        decoder layers."""
        return self.build_attn(self.num_dec_attn_heads)

    def build_attn(self, num_heads: int) -> MultiheadAttention:
        """Builds a :class:`MultiheadAttention`.

        The default implementations of :meth:`build_encoder_attn`,
        :meth:`build_decoder_attn`, and :meth:`build_encoder_decoder_attn`
        internally call this method.

        :param num_heads:
            The number of attention heads.
        """
        return StandardMultiheadAttention(
            num_heads, self.model_dim, batch_first=self.batch_first, **self._fct_kwargs
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Builds a :class:`FeedForwardNetwork`."""
        return StandardFeedForwardNetwork(
            self.model_dim, self.ffn_inner_dim, **self._fct_kwargs
        )
