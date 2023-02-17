# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter


@final
class Embedding(Module):
    """Stores embeddings of a fixed dictionary."""

    num_embed: int
    """The size of the dictionary."""

    embedding_dim: int
    """The dimensionality of returned embeddings."""

    padding_idx: Optional[int]
    """If not ``None``, entries at :attr:`padding_idx` do not contribute to the
    gradient; therefore, the embedding at :attr:`padding_idx` is not updated
    during training."""

    scaled: bool
    """If ``True``, the embeddings have been initialized from
    :math:`\\mathcal{N}(0, \\frac{1}{\\text{embedding_dim}})`; otherwise, from
    :math:`\\mathcal{N}(0, 1)`."""

    weight: Parameter
    """The learnable embeddings."""

    def __init__(
        self,
        num_embed: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        scaled: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        """
        :param num_embed:
            The size of the embedding dictionary.
        :param embedding_dim:
            The dimensionality of returned embeddings.
        :param padding_idx:
            If not ``None``, entries at ``padding_idx`` do not contribute to the
            gradient; therefore, the embedding at ``padding_idx`` is not updated
            during training.
        :param scaled:
            If ``True``, the embeddings will be initialized from
            :math:`\\mathcal{N}(0, \\frac{1}{\\text{embedding_dim}})`; otherwise,
            from :math:`\\mathcal{N}(0, 1)`.
        """
        super().__init__()

        self.num_embed = num_embed
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.scaled = scaled

        self.weight = Parameter(
            torch.empty((num_embed, embedding_dim), device=device, dtype=dtype)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.scaled:
            nn.init.normal_(self.weight, std=self.embedding_dim**-0.5)
        else:
            nn.init.normal_(self.weight)

        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input from which to extract the indices. *Shape:* Any.

        :returns:
            The embeddings. *Shape:* :math:`(*,E)`, where :math:`*` is the input
            shape and :math:`E` is the embedding size.
        """
        return F.embedding(x, self.weight, self.padding_idx)

    def extra_repr(self) -> str:
        """:meta private:"""
        s = f"num_embed={self.num_embed}, embedding_dim={self.embedding_dim}"

        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"

        if self.scaled:
            s += ", scaled=True"

        return s
