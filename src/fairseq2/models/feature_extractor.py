# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from torch import Tensor
from torch.nn import Module

from fairseq2.nn.padding import PaddingMask


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
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """
        :param seqs:
            The sequences from which to extract features. *Shape:*
            :math:`(N,S,*)`, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :returns:
            - The extracted features. *Shape:* :math:`(N,S_{out},F)`, where
              :math:`N` is the batch size, :math:`S_{out}` is the output
              sequence length, and :math:`F` is the dimensionality of the
              features.
            - The padding mask of the extracted features. *Shape:*
              :math:`(N,S_{out})`, where :math:`N` is the batch size and
              :math:`S_{out}` is the output sequence length.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"feature_dim={self.feature_dim}"
