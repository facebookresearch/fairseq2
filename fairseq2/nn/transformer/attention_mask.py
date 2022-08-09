# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NoReturn, Optional, Protocol, final

from torch import Tensor


class AttentionMaskGenerator(Protocol):
    """Generates an attention mask."""

    def __call__(self, tgt: Tensor, batch_first: bool = False) -> Tensor:
        """
        :param tgt:
            The target for which to generate the mask. *Shape:* :math:`(S,*)`
            when unbatched, :math:`(S,N,*)` when ``batch_first`` is ``True``, or
            :math:`(N,S,*)` when ``batch_first`` is ``False``, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param batch_first:
            If ``True``, the first dimension of ``tgt`` represents the batch;
            otherwise, the sequence.

        :returns:
            An attention mask whose content is specific to the generator.
            *Shape:* :math:`(S,S)`, where :math:`S` is the sequence length.
        """


@final
class CausalAttentionMaskGenerator:
    """Generates a causal attention mask for self attention."""

    _cached_attn_mask: Optional[Tensor]

    def __init__(self) -> None:
        self._cached_attn_mask = None

    def __call__(self, tgt: Tensor, batch_first: bool = False) -> Tensor:
        """
        :param tgt:
            The target for which to generate the mask. *Shape:* :math:`(S,*)`
            when unbatched, :math:`(S,N,*)` when ``batch_first`` is ``True``, or
            :math:`(N,S,*)` when ``batch_first`` is ``False``, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param batch_first:
            If ``True``, the first dimension of ``tgt`` represents the batch;
            otherwise, the sequence.

        :returns:
            An attention mask whose upper triangular part above the main
            diagonal is filled with negative infinities (i.e. ``float("-inf")``)
            while its rest is filled with zeros. *Shape:* :math:`(S,S)`, where
            :math:`S` is the sequence length.

        Usage:

        >>> import torch
        >>>
        >>> from fairseq2.modules.tranformer import CausalAttentionMaskGenerator
        >>>
        >>> g = CausalAttentionMaskGenerator()
        >>> g(torch.empty(4, 10))
        tensor([[0., -inf, -inf, -inf],
                [0.,   0., -inf, -inf],
                [0.,   0.,   0., -inf],
                [0.,   0.,   0.,   0.]])
        """
        mask = self._cached_attn_mask

        if batch_first and tgt.dim() > 1:
            seq_len = tgt.size(1)
        else:
            seq_len = tgt.size(0)

        if mask is None or mask.device != tgt.device or mask.size(0) < seq_len:
            mask = tgt.new_full([seq_len, seq_len], float("-inf"))

            mask.triu_(diagonal=1)

            self._cached_attn_mask = mask

        return mask[:seq_len, :seq_len]


@final
class ALiBiAttentionMaskGenerator:
    """Generates a mask for self attention as described in
    :cite:t:`DBLP:journals/corr/abs-2108-12409`.

    .. todo:: Not implemented yet!
    """

    def __call__(self, tgt: Tensor, batch_first: bool = False) -> NoReturn:
        raise NotImplementedError()
