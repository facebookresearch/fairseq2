# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import embedding
from torch.nn.parameter import Parameter

from fairseq2.typing import DataType, Device


@final
class Embedding(Module):
    """Stores embeddings of a fixed dictionary and size."""

    num_embeddings: int
    embedding_dim: int
    pad_idx: Optional[int]
    padding_idx: Optional[int]  # Compat
    scaled: bool
    weight: Parameter

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        pad_idx: Optional[int] = None,
        scaled: bool = False,
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
        :param scaled:
            If ``True``, initializes the embeddings from
            :math:`\\mathcal{N}(0, \\frac{1}{\\text{embedding_dim}})`; otherwise,
            from :math:`\\mathcal{N}(0, 1)`.
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.pad_idx = pad_idx
        self.scaled = scaled

        # Alias field for compatibility with `torch.nn.Embedding`.
        self.padding_idx = pad_idx

        self.weight = Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.scaled:
            nn.init.normal_(self.weight, std=self.embedding_dim**-0.5)
        else:
            nn.init.normal_(self.weight)

        if self.pad_idx is not None:
            with torch.no_grad():
                self.weight[self.pad_idx].fill_(0.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:
            The embedding indices. *Shape:* Any.

        :returns:
            The embeddings corresponding to the specified indices. *Shape:*
            :math:`(*,E)`, where :math:`*` is the input shape and :math:`E` is
            the dimensionality of the embeddings.
        """
        return embedding(x, self.weight, self.pad_idx)

    def extra_repr(self) -> str:
        """:meta private:"""
        s = f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"

        if self.pad_idx is not None:
            s += f", pad_idx={self.pad_idx}"

        if self.scaled:
            s += ", scaled=True"

        return s
