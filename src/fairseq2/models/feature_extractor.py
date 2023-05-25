# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from torch import Tensor
from torch.nn import Module


class SequenceFeatureExtractor(Module, ABC):
    """Extracts features from sequences and embeds them in a latent space."""

    feature_dim: int

    def __init__(self, feature_dim: int) -> None:
        """
        :param feature_dim:
            The dimensionality of extracted features.
        """
        super().__init__()

        self.feature_dim = feature_dim

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
            - The extracted features. *Shape:* :math:`(N,S_{out},F)`, where
              :math:`N` is the batch size, :math:`S_{out}` is the output
              sequence length, and :math:`F` is the dimensionality of the
              features.
            - An array where each element represents the length of the sequence
              at the same index in the extracted features. *Shape:* :math:`(N)`,
              where :math:`N` is the batch size.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"feature_dim={self.feature_dim}"
