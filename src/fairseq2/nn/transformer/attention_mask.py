# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NoReturn, Optional, Protocol, final

from torch import Tensor

from fairseq2.nn.utils import neg_inf


class AttentionMaskGenerator(Protocol):
    """Generates an attention mask."""

    def __call__(self, tgt: Tensor) -> Tensor:
        """
        :param tgt:
            The target for which to generate the mask. *Shape:* :math:`(N,S,*)`,
            or :math:`(S,*)` when unbatched, where :math:`N` is the batch size
            and :math:`S` is the sequence length.

        :returns:
            An attention mask whose content is specific to the generator.
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

    def __call__(self, tgt: Tensor) -> Tensor:
        """
        :param tgt:
            The target for which to generate the mask. *Shape:* :math:`(N,S,*)`,
            or :math:`(S,*)` when unbatched, where :math:`N` is the batch size
            and :math:`S` is the sequence length.

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
        >>> g(torch.empty(4, 10))
        tensor([[0., -inf, -inf, -inf],
                [0.,   0., -inf, -inf],
                [0.,   0.,   0., -inf],
                [0.,   0.,   0.,   0.]])
        """
        mask = self._cached_attn_mask

        if tgt.dim() > 1:
            seq_len = tgt.size(1)
        else:
            seq_len = tgt.size(0)

        if mask is None or mask.device != tgt.device or mask.size(0) < seq_len:
            mask = tgt.new_full([seq_len, seq_len], neg_inf)

            mask.triu_(diagonal=1)

            self._cached_attn_mask = mask

        return mask[:seq_len, :seq_len]


@final
class ALiBiAttentionMaskGenerator:
    """Generates a mask for self attention as described in
    :cite:t:`DBLP:journals/corr/abs-2108-12409`.

    .. note::
        This class follows the :class:`AttentionMaskGenerator` protocol.

    .. todo:: Not implemented yet!
    """

    def __call__(self, tgt: Tensor) -> NoReturn:
        raise NotImplementedError()
