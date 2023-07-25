# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Protocol

import torch
from torch import Tensor


class AttentionMaskGenerator(Protocol):
    """Generates an attention mask."""

    def __call__(self, seqs: Tensor) -> Tensor:
        """
        :param seqs:
            The sequences for which to generate the mask. *Shape:*
            :math:`(N,S,M)`, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`M` is the dimensionality of the model.

        :returns:
            An implementation-defined attention mask for ``seqs`` specific to
            the generator. *Shape:* :math:`(S,S)`, where :math:`S` is the
            sequence length.
        """


class CausalAttentionMaskGenerator:
    """Generates a causal attention mask for self attention.

    .. note::
        This class follows the :class:`AttentionMaskGenerator` protocol.
    """

    _cached_attn_mask: Optional[Tensor]

    def __init__(self) -> None:
        self._cached_attn_mask = None

    def __call__(self, seqs: Tensor) -> Tensor:
        """
        :param seqs:
            The sequences for which to generate the mask. *Shape:*
            :math:`(N,S,M)`, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`M` is the dimensionality of the model.

        :returns:
            An attention mask for ``seqs`` whose upper triangular part above the
            main diagonal is filled with negative infinities while its rest is
            filled with zeros. *Shape:* :math:`(S,S)`, where :math:`S` is the
            sequence length.

        Usage:

        >>> import torch
        >>>
        >>> from fairseq2.nn.transformer import CausalAttentionMaskGenerator
        >>>
        >>> g = CausalAttentionMaskGenerator()
        >>> g(torch.empty(4, 10, 3))
        tensor([[0., -inf, -inf, -inf],
                [0.,   0., -inf, -inf],
                [0.,   0.,   0., -inf],
                [0.,   0.,   0.,   0.]])
        """
        mask = self._cached_attn_mask

        seq_len = seqs.size(1)

        if mask is None or mask.device != seqs.device or mask.size(0) < seq_len:
            mask = seqs.new_full([seq_len, seq_len], -torch.inf)

            mask.triu_(diagonal=1)

            self._cached_attn_mask = mask

        # The `is_causal` tag is checked by efficient SDPA implementations to
        # optimize attention masking.
        setattr(mask, "is_causal", True)

        return mask[:seq_len, :seq_len]

    def __repr__(self) -> str:
        return "CausalAttentionMaskGenerator"


class ALiBiAttentionMaskGenerator:
    """Generates a mask for self attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2108.12409`.

    .. note::
        This class follows the :class:`AttentionMaskGenerator` protocol.
    """

    _cached_attn_mask: Optional[Tensor]
    _causal_attn_mask_gen: CausalAttentionMaskGenerator

    def __init__(self, num_heads: int) -> None:
        self.num_heads = num_heads

        self._cached_attn_mask = None
        self._causal_attn_mask_gen = CausalAttentionMaskGenerator()

    def __call__(self, seqs: Tensor) -> Tensor:
        """
        :param seqs:
            The sequences for which to generate the mask. *Shape:*
            :math:`(N,S,M)`, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`M` is the dimensionality of the model.

        :returns:
            An ALiBi mask for ``seqs``. *Shape:* :math:`(H,S,S)`, where :math:`S`
            is the sequence length and :math:`H` is the number of attention
            heads.
        """
        mask = self._cached_attn_mask

        seq_len = seqs.size(1)

        if mask is None or mask.device != seqs.device or mask.size(1) < seq_len:
            slopes = self._get_slopes(self.num_heads)

            arange_tensor = torch.arange(seq_len, device=seqs.device)[None, None, :]
            arange_tensor = arange_tensor.expand((self.num_heads, -1, -1))

            alibi_biases = arange_tensor * slopes[:, None, None]
            mask = alibi_biases + self._causal_attn_mask_gen(seqs)

            self._cached_attn_mask = mask

        return mask[:, :seq_len, :seq_len]

    def _get_slopes(self, num_heads: int) -> Tensor:
        def get_slopes_power_of_2(num_heads: int, step: int = 1) -> Tensor:
            start = 2 ** (-8 / num_heads)
            return torch.pow(start, torch.arange(1, 1 + num_heads, step))

        num_heads_log_2 = math.log2(num_heads)
        if num_heads_log_2.is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_pow_2 = 2 ** math.floor(num_heads_log_2)
            base_slopes = get_slopes_power_of_2(closest_pow_2)
            num_slopes_left = num_heads - closest_pow_2
            extra_slopes = get_slopes_power_of_2(2 * closest_pow_2, step=2)

            return torch.cat([base_slopes, extra_slopes[:num_slopes_left]])

    def __repr__(self) -> str:
        return "ALiBiAttentionMaskGenerator"
