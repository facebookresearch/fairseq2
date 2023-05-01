# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from torch import Tensor
from torch.nn import Module


class FeatureExtractor(Module, ABC):
    """Extracts features from inputs and embeds them in a latent space."""

    embed_dim: int

    def __init__(self, embed_dim: int) -> None:
        """
        :param embed_dim:
            The dimensionality of returned embeddings.
        """
        super().__init__()

        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param seqs:
            The sequences from which to extract features. *Shape:*
            :math:`(N,S,*)`, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.

        :returns:
            - The embeddings extracted from ``seqs``. *Shape:*
              :math:`(N,S_{out},E)`, where :math:`N` is the batch size,
              :math:`S_{out}` is the output sequence length, and :math:`E` is
              the embedding size.
            - An array where each element represents the length of the sequence
              at the same index in the returned embeddings. *Shape:*
              :math:`(N)`, where :math:`N` is the batch size.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"embed_dim={self.embed_dim}"
