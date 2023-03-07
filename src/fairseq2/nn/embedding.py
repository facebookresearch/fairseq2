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
    embed_dim: int
    embedding_dim: int  # Compat
    pad_idx: Optional[int]
    padding_idx: Optional[int]  # Compat
    scaled: bool
    weight: Parameter

    def __init__(
        self,
        num_embed: int,
        embed_dim: int,
        pad_idx: Optional[int] = None,
        scaled: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param num_embed:
            The size of the embedding dictionary.
        :param embed_dim:
            The dimensionality of returned embeddings.
        :param pad_idx:
            If not ``None``, entries at ``pad_idx`` do not contribute to the
            gradient; therefore, the embedding at ``pad_idx`` is not updated
            during training.
        :param scaled:
            If ``True``, the embeddings will be initialized from
            :math:`\\mathcal{N}(0, \\frac{1}{\\text{embed_dim}})`; otherwise,
            from :math:`\\mathcal{N}(0, 1)`.
        """
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx
        self.scaled = scaled

        # Alias fields for compatibility with torch.nn.Embedding.
        self.embedding_dim = embed_dim
        self.padding_idx = pad_idx

        self.weight = Parameter(
            torch.empty((num_embed, embed_dim), device=device, dtype=dtype)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""
        if self.scaled:
            nn.init.normal_(self.weight, std=self.embed_dim**-0.5)
        else:
            nn.init.normal_(self.weight)

        if self.pad_idx is not None:
            with torch.no_grad():
                self.weight[self.pad_idx].fill_(0.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input from which to extract the indices. *Shape:* Any.

        :returns:
            The embeddings. *Shape:* :math:`(*,E)`, where :math:`*` is the input
            shape and :math:`E` is the embedding size.
        """
        return F.embedding(x, self.weight, self.pad_idx)

    def extra_repr(self) -> str:
        """:meta private:"""
        s = f"num_embed={self.num_embed}, embed_dim={self.embed_dim}"

        if self.pad_idx is not None:
            s += f", pad_idx={self.pad_idx}"

        if self.scaled:
            s += ", scaled=True"

        return s
