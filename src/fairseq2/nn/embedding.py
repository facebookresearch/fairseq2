# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import embedding
from torch.nn.parameter import Parameter

from fairseq2.typing import DataType, Device, finaloverride


class Embedding(Module, ABC):
    """Stores embeddings of a fixed dictionary and size."""

    num_embeddings: int
    embedding_dim: int
    pad_idx: Optional[int]
    padding_idx: Optional[int]  # Compat

    def __init__(
        self, num_embeddings: int, embedding_dim: int, pad_idx: Optional[int] = None
    ) -> None:
        """
        :param num_embeddings:
            The size of the embedding table.
        :param embedding_dim:
            The dimensionality of returned embeddings.
        :param pad_idx:
            If not ``None``, entries at ``pad_idx`` do not contribute to the
            gradient; therefore, the embedding at ``pad_idx`` is not updated
            during training.
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.pad_idx = pad_idx

        # Alias field for compatibility with `torch.nn.Embedding`.
        self.padding_idx = pad_idx

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:
            The embedding indices. *Shape:* Any.

        :returns:
            The embeddings corresponding to the specified indices. *Shape:*
            :math:`(*,E)`, where :math:`*` is the input shape and :math:`E` is
            the dimensionality of the embeddings.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        s = f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"

        if self.pad_idx is not None:
            s = f"{s}, pad_idx={self.pad_idx}"

        return s


@final
class StandardEmbedding(Embedding):
    """Stores embeddings of a fixed dictionary and size in an in-memory table."""

    weight: Parameter
    init_fn: Optional[Callable[[StandardEmbedding], None]]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        pad_idx: Optional[int] = None,
        *,
        init_fn: Optional[Callable[[StandardEmbedding], None]] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param num_embeddings:
            The size of the embedding table.
        :param embedding_dim:
            The dimensionality of returned embeddings.
        :param pad_idx:
            If not ``None``, entries at ``pad_idx`` do not contribute to the
            gradient; therefore, the embedding at ``pad_idx`` is not updated
            during training.
        :param init_fn:
            The callable to use for parameter initialization.
        """
        super().__init__(num_embeddings, embedding_dim, pad_idx)

        self.weight = Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.init_fn is not None:
            self.init_fn(self)

            return

        nn.init.normal_(self.weight)

        if self.pad_idx is not None:
            with torch.no_grad():
                self.weight[self.pad_idx].fill_(0.0)

    @finaloverride
    def forward(self, x: Tensor) -> Tensor:
        return embedding(x, self.weight, self.pad_idx)

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.init_fn is not None:
            init_fn = getattr(self.init_fn, "__name__", self.init_fn)

            s = f"{s}, init_fn={init_fn}"

        return s


def init_scaled_embedding(embed: StandardEmbedding) -> None:
    """Initialize ``embed`` from
    :math:`\\mathcal{N}(0, \\frac{1}{\\text{embedding_dim}})`."""
    nn.init.normal_(embed.weight, std=embed.embedding_dim**-0.5)

    if embed.pad_idx is not None:
        with torch.no_grad():
            embed.weight[embed.pad_idx].fill_(0.0)
