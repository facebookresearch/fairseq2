# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch import Tensor
from torch.nn import Module

from fairseq2.models.transformer.attention_bias import AttentionBiasCache
from fairseq2.nn import BatchLayout


class SDPA(Module, ABC):
    """Computes scaled dot-product attention."""

    @abstractmethod
    def forward(
        self,
        q: Tensor,
        q_layout: BatchLayout,
        k: Tensor,
        k_layout: BatchLayout,
        v: Tensor,
        bias_cache: AttentionBiasCache,
        *,
        needs_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        :param q: The sequences to query. *Shape:* :math:`(N,S,H,K)`, where
            :math:`N` is the batch size, :math:`H` is the number of heads,
            :math:`S` is the sequence length, and :math:`K` is the key size.
        :param k: The keys. *Shape:* :math:`(N,S_{kv},H,K)`, where :math:`N`
            is the batch size, :math:`H` is the number of heads, :math:`S_{kv}`
            is the key/value sequence length, and :math:`K` is the key size.
        :param v: The values. *Shape:* :math:`(N,S_{kv},H,V)`, where
            :math:`N` is the batch size, :math:`H` is the number of heads,
            :math:`S_{kv}` is the key/value sequence length, and :math:`V` is
            the value size.
        :param needs_weights: If ``True``, returns the attention weights.

        :returns:
            - The attention values. *Shape:* :math:`(N,S,H,V)`, where :math:`N`
              is the batch size, :math:`H` is the number of heads, :math:`S` is
              the sequence length, and :math:`V` is the value size.
            - The attention weights. *Shape:* :math:`(N,H,S,S_{kv})`, where
              :math:`N` is the batch size, :math:`H` is the number of heads,
              :math:`S` is the sequence length, and :math:`S_{kv}` is the
              key/value sequence length.
        """

    if TYPE_CHECKING:
        __call__ = forward
