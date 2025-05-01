# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module

from fairseq2.nn import BatchLayout


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
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> tuple[Tensor, BatchLayout]:
        """
        :param seqs:
            The sequences from which to extract features. *Shape:*
            :math:`(N,S,*)`, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.

        :returns:
            - The extracted features. *Shape:* :math:`(N,S_{out},E)`, where
              :math:`N` is the batch size, :math:`S_{out}` is the output
              sequence length, and :math:`E` is the dimensionality of the
              features.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"feature_dim={self.feature_dim}"
