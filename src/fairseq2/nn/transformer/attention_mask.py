# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NoReturn, Optional, Protocol, final

from torch import Tensor

_neg_inf = float("-inf")


class AttentionMaskGenerator(Protocol):
    """Generates an attention mask."""

    def __call__(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input for which to generate the mask. *Shape:* :math:`(N,S,M)`,
            or :math:`(S,M)` when unbatched, where :math:`N` is the batch size,
            :math:`S` is the sequence length, and :math:`M` is the model size.

        :returns:
            An implementation-defined attention mask specific to the generator.
            *Shape:* :math:`(S,S)`, where :math:`S` is the sequence length.
        """


@final
class CausalAttentionMaskGenerator:
    """Generates a causal attention mask for self attention.

    .. note::
        This class follows the :class:`AttentionMaskGenerator` protocol.
    """

    _cached_attn_mask: Optional[Tensor]

    def __init__(self) -> None:
        self._cached_attn_mask = None

    def __call__(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input for which to generate the mask. *Shape:* :math:`(N,S,M)`,
            or :math:`(S,M)` when unbatched, where :math:`N` is the batch size,
            :math:`S` is the sequence length, and :math:`M` is the model size.

        :returns:
            An attention mask whose upper triangular part above the main
            diagonal is filled with negative infinity (i.e. ``float("-inf")``),
            while its rest is filled with zero. *Shape:* :math:`(S,S)`, where
            :math:`S` is the sequence length.

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

        if x.dim() == 2:
            seq_len = x.size(0)
        else:
            seq_len = x.size(1)

        if mask is None or mask.device != x.device or mask.size(0) < seq_len:
            mask = x.new_full([seq_len, seq_len], _neg_inf)

            mask.triu_(diagonal=1)

            self._cached_attn_mask = mask

        return mask[:seq_len, :seq_len]


@final
class ALiBiAttentionMaskGenerator:
    """Generates a mask for self attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2108.12409`.

    .. note::
        This class follows the :class:`AttentionMaskGenerator` protocol.

    .. todo:: Not implemented yet!
    """

    def __call__(self, x: Tensor) -> NoReturn:
        raise NotImplementedError()
