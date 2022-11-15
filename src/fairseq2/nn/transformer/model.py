# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional, final

import torch.nn as nn
import torch.nn.functional as F
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Module

from fairseq2.nn.projection import Projection, ResettableProjection
from fairseq2.nn.transformer.decoder import TransformerDecoder
from fairseq2.nn.transformer.encoder import TransformerEncoder
from fairseq2.typing import DataType, Device


class Transformer(Module, ABC):
    """Represents a Transformer model."""

    model_dim: int
    """The dimensionality of the model (i.e. inputs and outputs)."""

    batch_first: bool
    """If ``True``, the first dimension of batched inputs and outputs represents
    the batch; otherwise, the sequence."""

    encoder: TransformerEncoder
    decoder: TransformerDecoder
    score_proj: Projection

    def __init__(self, model_dim: int, batch_first: bool) -> None:
        """
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs).
        :param batch_first:
            If ``True``, the first dimension of batched inputs and outputs
            represents the batch; otherwise, the sequence.
        """
        super().__init__()

        self.model_dim = model_dim

        self.batch_first = batch_first

    @abstractmethod
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

    @abstractmethod
    def extract_features(self, src_seq: Tensor, tgt_seq: Tensor) -> Tensor:
        """
        :returns: Features before the softmax layer
        """


@final
class StandardTransformer(Transformer):
    """Represents a Transformer model as described in
    :cite:t:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    """

    use_log_softmax: bool

    def __init__(
        self,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        score_proj: Projection,
        use_log_softmax: bool = False,
    ) -> None:
        """
        :param encoder:
            The encoder.
        :param decoder:
            The decoder.
        :param score_proj:
            The projection to apply to the outputs of the decoder.
        :param use_log_softmax:
            If ``True``, apply log-softmax instead of softmax to the scores
            (i.e. logits) produced by ``score_proj``.
        """
        if encoder.model_dim != decoder.model_dim:
            raise ValueError(
                f"`model_dim` of `encoder` ({encoder.model_dim}) does not match `model_dim` of `decoder` ({decoder.model_dim})."
            )

        if encoder.batch_first != decoder.batch_first:
            raise ValueError(
                f"`batch_first` of `encoder` ({encoder.batch_first}) does not match `batch_first` of `decoder` ({decoder.batch_first})."
            )

        super().__init__(encoder.model_dim, encoder.batch_first)

        self.encoder = encoder
        self.decoder = decoder
        self.score_proj = score_proj

        self.use_log_softmax = use_log_softmax

    @finaloverride
    def forward(self, src_seq: Tensor, tgt_seq: Tensor) -> Tensor:
        x = self.extract_features(src_seq, tgt_seq)
        softmax = F.log_softmax if self.use_log_softmax else F.softmax
        return softmax(x, dim=-1)

    @finaloverride
    def extract_features(self, src_seq: Tensor, tgt_seq: Tensor) -> Tensor:
        enc_out, enc_attn_padding_mask = self.encoder(src_seq)
        x = self.decoder(tgt_seq, enc_out, enc_attn_padding_mask)
        x = self.score_proj(x)
        return x  # type: ignore


@final
class UntiedScoreProjection(ResettableProjection):
    """Produces scores -i.e. logits- from the output of a Transformer decoder.

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
