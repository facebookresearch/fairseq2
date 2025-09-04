# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final

from torch import Tensor
from torch.nn import Dropout, Module
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn import BatchLayout, Embedding, IncrementalStateBag, PositionEncoder


class TransformerFrontend(Module, ABC):
    """Represents a Transformer encoder/decoder front-end."""

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> tuple[Tensor, BatchLayout]:
        """
        :param seqs:
            The sequences to process. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            - The processed sequences to pass to a Transformer encoder/decoder.
              *Shape:* :math:`(N,S_{out},M)`, where :math:`N` is the batch size,
              :math:`S_{out}` is the output sequence length, and :math:`M` is
              the dimensionality of the model.
        """

    if TYPE_CHECKING:
        __call__ = forward


@final
class TransformerEmbeddingFrontend(TransformerFrontend):
    """Represents a Transformer encoder/decoder front-end as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    def __init__(
        self,
        model_dim: int,
        embed: Embedding,
        pos_encoder: PositionEncoder | None,
        *,
        no_scale: bool = False,
        dropout_p: float = 0.0,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param embed:
            The token embedding table.
        :param pos_encoder:
            The position encoder.
        :param no_scale:
            If ``True``, does not scale embeddings by the square root of the
            embedding size.
        :param layer_norm:
            If ``True``, applies Layer Normalization to embeddings before
            dropout.
        :param dropout_p:
            The dropout probability on embeddings.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization module.
        """
        super().__init__()

        self.embed = embed

        self.scale = 1.0 if no_scale else math.sqrt(model_dim)

        self.pos_encoder: PositionEncoder | None

        self.register_module("pos_encoder", pos_encoder)

        if dropout_p > 0.0:
            dropout = Dropout(dropout_p)
        else:
            dropout = None

        self.dropout: Dropout | None

        self.register_module("dropout", dropout)

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> tuple[Tensor, BatchLayout]:
        seqs = self.embed(seqs)

        if self.scale != 1.0:
            seqs = seqs * self.scale

        if self.pos_encoder is not None:
            seqs = self.pos_encoder(seqs, seqs_layout, state_bag=state_bag)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs, seqs_layout

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        if self.scale != 1.0:
            return f"no_scale={False}"

        return ""
