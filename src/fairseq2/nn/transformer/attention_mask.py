# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from threading import Lock
from typing import Dict, Optional, Protocol, final

import torch
from torch import Tensor

from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import DataType, Device, finaloverride


class AttentionMask(ABC):
    """Represents an attention mask."""

    materialized: Optional[Tensor]
    """The attention mask tensor. Will be ``None`` till the first call to
    :method:`materialize`."""

    def __init__(self) -> None:
        self.materialized = None

    def materialize(self) -> Tensor:
        """Materialize the attention mask tensor."""
        if self.materialized is None:
            self.materialized = self._do_materialize()

        return self.materialized

    @abstractmethod
    def _do_materialize(self) -> Tensor:
        ...


class AttentionMaskFactory(Protocol):
    """Constructs an attention mask."""

    def __call__(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        training: bool = True,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Optional[AttentionMask]:
        """
        :param seqs:
            The sequences for which to create a mask. *Shape:* :math:`(*,S,M)`,
            where :math:`*` is any number of batch dimensions including none,
            :math:`S` is the sequence length, and :math:`M` is the
            dimensionality of the model.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param training:
            If ``True``, indicates that the calling module is in training mode.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            An implementation-defined mask for ``seqs``.
        """


@final
class CustomAttentionMask(AttentionMask):
    """Represents a custom attention mask."""

    def __init__(self, mask: Tensor) -> None:
        """
        :param mask:
            The custom attention mask tensor.
        """
        super().__init__()

        self.mask = mask

    @finaloverride
    def _do_materialize(self) -> Tensor:
        return self.mask


@final
class GlobalCausalAttentionMask(AttentionMask):
    """Represents a global causal attention mask.

    *Shape:* :math:`(S,S)`, where :math:`S` is the sequence length.

    Usage:

    >>> import torch
    >>>
    >>> from fairseq2.nn.transformer import GlobalCausalAttentionMask
    >>>
    >>> mask = GlobalCausalAttentionMask(torch.empty(4, 10, 3))
    >>> mask.materialize()
    tensor([[0., -inf, -inf, -inf],
            [0.,   0., -inf, -inf],
            [0.,   0.,   0., -inf],
            [0.,   0.,   0.,   0.]])
    """

    _cache_lock = Lock()

    _cache_mask: Optional[Tensor] = None

    def __init__(self, seq_len: int, device: Device, dtype: DataType) -> None:
        """
        :param seq_len:
            The sequence length of the mask.
        """
        super().__init__()

        self.seq_len = seq_len

        self.device, self.dtype = device, dtype

    @finaloverride
    def _do_materialize(self) -> Tensor:
        kls = GlobalCausalAttentionMask

        with kls._cache_lock:
            if self._should_update_cache():
                kls._cache_mask = _create_causal_attention_mask(
                    self.seq_len, self.device, self.dtype
                )
            else:
                assert kls._cache_mask is not None

            return kls._cache_mask[: self.seq_len, : self.seq_len]

    def _should_update_cache(self) -> bool:
        mask = GlobalCausalAttentionMask._cache_mask

        if mask is None:
            return True

        if mask.dtype != self.dtype or mask.device != self.device:
            return True

        return mask.size(1) < self.seq_len


class GlobalCausalAttentionMaskFactory:
    def __call__(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        training: bool = True,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Optional[GlobalCausalAttentionMask]:
        seq_len = seqs.size(-2)

        if seq_len <= 1:
            # Return `None` if this is the first step; or if we attend to all
            # past steps in incremental decoding.
            return None

        return GlobalCausalAttentionMask(seq_len, seqs.device, seqs.dtype)

    def __repr__(self) -> str:
        return "GlobalCausalAttentionMaskFactory()"


@final
class ALiBiMask(AttentionMask):
    """Represents an ALiBi attention mask as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2108.12409`.

    *Shape:* :math:`(H,S,S)`, where :math:`H` is the number of attention heads
    and :math:`S` is the sequence length.
    """

    _cache_lock = Lock()

    _cache_masks: Dict[int, Tensor] = {}

    def __init__(
        self, seq_len: int, num_heads: int, device: Device, dtype: DataType
    ) -> None:
        """
        :param seq_len:
            The sequence length of the mask.
        :param num_heads:
            The number of attention heads.
        """
        super().__init__()

        if num_heads % 2 != 0:
            raise ValueError(f"`num_heads` must be even, but is {num_heads} instead.")

        self.seq_len = seq_len
        self.num_heads = num_heads

        self.device, self.dtype = device, dtype

    @finaloverride
    def _do_materialize(self) -> Tensor:
        kls = ALiBiMask

        with kls._cache_lock:
            if self._should_update_cache():
                kls._cache_masks[self.num_heads] = self._init_cache_mask()
            else:
                assert kls._cache_masks is not None

            return kls._cache_masks[self.num_heads][:, : self.seq_len, : self.seq_len]

    def _should_update_cache(self) -> bool:
        mask = ALiBiMask._cache_masks.get(self.num_heads, None)

        if mask is None:
            return True

        if mask.dtype != self.dtype or mask.device != self.device:
            return True

        return mask.size(1) < self.seq_len

    def _init_cache_mask(self) -> Tensor:
        # (S, S)
        causal_mask = _create_causal_attention_mask(
            self.seq_len, self.device, self.dtype
        )

        # (H)
        powers = torch.arange(1, 1 + self.num_heads, device=self.device)

        # (H)
        slopes = torch.pow(2 ** (-8 / self.num_heads), powers)

        # (S)
        steps = torch.arange(self.seq_len, device=self.device)

        # (S) -> (H, 1, S)
        steps = steps[None, None, :].expand(self.num_heads, -1, -1)

        # (H, 1, S) * (H, 1, 1) -> (H, 1, S)
        biases = steps * slopes[:, None, None]

        # (H, 1, S) + (S, S) -> (H, S, S)
        mask = biases.to(self.dtype) + causal_mask

        return mask


class ALiBiMaskFactory:
    num_heads: int

    def __init__(self, num_heads: int) -> None:
        """
        :param num_heads:
            The number of attention heads.
        """
        self.num_heads = num_heads

    def __call__(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        training: bool = True,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Optional[ALiBiMask]:
        if not training and state_bag is not None:
            start = state_bag.step
        else:
            start = 0

        seq_len = start + seqs.size(-2)

        if seq_len <= 1:
            return None  # Nothing to attend to.

        return ALiBiMask(seq_len, self.num_heads, seqs.device, seqs.dtype)

    def __repr__(self) -> str:
        return f"ALiBiMaskFactory(num_heads={self.num_heads})"


def _create_causal_attention_mask(
    seq_len: int, device: Device, dtype: DataType
) -> Tensor:
    # As of PyTorch 2.0, `triu` does not support bf16.
    dt = torch.float32 if dtype == torch.bfloat16 else dtype

    mask = torch.full((seq_len, seq_len), -torch.inf, device=device, dtype=dt)

    mask.triu_(diagonal=1)

    return mask.to(dtype)
