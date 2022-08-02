# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import final

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from ..projection import Projection
from .decoder import TransformerDecoder
from .encoder import TransformerEncoder


class Transformer(Module, ABC):
    """Represents a Transformer model.

    :param model_dim:
        The dimensionality of the model (i.e. inputs and outputs).
    :param batch_first:
        If ``True``, the first dimension of the batched inputs and outputs
        represents the batch; otherwise, the sequence.
    """

    model_dim: int
    """The dimensionality of the model (i.e. inputs and outputs)."""

    batch_first: bool
    """If ``True``, the first dimension of the batched inputs and outputs
    represents the batch; otherwise, the sequence."""

    def __init__(self, model_dim: int, batch_first: bool) -> None:
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


@final
class StandardTransformer(Transformer):
    """Represents a Transformer model as described in
    :cite:t:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    :param encoder:
        The encoder.
    :param decoder:
        The decoder.
    :param dict_proj:
        The dictionary projection to apply to the output of the decoder.
    :param log_out_probs:
        If ``True``, apply log-softmax instead of softmax to the output of the
        dictionary projection.
    """

    encoder: TransformerEncoder
    decoder: TransformerDecoder
    dict_proj: Projection
    log_out_probs: bool

    def __init__(
        self,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        dict_proj: Projection,
        log_out_probs: bool = False,
    ) -> None:
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

        self.dict_proj = dict_proj

        self.log_out_probs = log_out_probs

    def forward(self, src_seq: Tensor, tgt_seq: Tensor) -> Tensor:  # override
        enc_out, enc_attn_padding_mask = self.encoder(src_seq)

        x = self.decoder(tgt_seq, enc_out, enc_attn_padding_mask)

        x = self.dict_proj(x)

        if self.log_out_probs:
            softmax = F.log_softmax
        else:
            softmax = F.softmax

        return softmax(x, dim=-1)
