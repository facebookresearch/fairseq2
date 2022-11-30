# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, final

import torch.nn as nn
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Module

from fairseq2.nn.projection import Projection, ResettableProjection
from fairseq2.nn.transformer.decoder import TransformerDecoder
from fairseq2.nn.transformer.encoder import TransformerEncoder
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
            The predicted next-token probabilities. *Shape:* :math:`(T,D)` when
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


@final
class UntiedScoreProjection(ResettableProjection):
    """Produces scores (i.e. logits) from the output of a Transformer decoder.

    The produced scores should be forwarded to a softmax function to compute
    predicted next-token probabilities.
    """

    def __init__(
        self,
        num_embed: int,
        embedding_dim: int,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param num_embed:
            The size of the output embedding dictionary.
        :param embedding_dim:
            The dimensionality of output embeddings.
        """
        super().__init__(
            embedding_dim, num_embed, bias=False, device=device, dtype=dtype
        )

    @finaloverride
    def reset_parameters(self) -> None:
        """Resets the parameters and buffers of the module."""
        nn.init.normal_(self.weight, std=self.inp_dim**-0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias)
