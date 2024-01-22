# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional, Protocol, final

import torch
from torch import Tensor

from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.typing import DataType, Device, finaloverride


class AttentionMask(ABC):
    """Represents an attention mask."""

    materialized: Optional[Tensor]

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
    """Constructs instances of :class:`AttentionMask`."""

    def __call__(
        self,
        seqs: Tensor,
        keys: Tensor,
        *,
        training: bool = True,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Optional[AttentionMask]:
        """
        :param seqs:
            The sequences for which to create a mask. *Shape:* :math:`(N,S,M)`,
            where :math:`N` is the batch size, :math:`S` is the sequence length,
            and :math:`M` is the dimensionality of the model.
        :param keys:
            The keys. *Shape:* :math:`(N,S_{kv},K)`, where :math:`N` is the
            batch size, :math:`S_{kv}` is the key/value sequence length, and
            :math:`K` is the key size.
        :param training:
            If ``True``, the calling module is in training mode.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            An implementation-defined mask for ``seqs``.
        """


@final
class CustomAttentionMask(AttentionMask):
    """Represents a custom attention mask provided by the user."""

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
class CausalAttentionMask(AttentionMask):
    """Represents a causal attention mask.

    *Shape:* :math:`(S,S_{kv})`, where :math:`S` is the sequence length and
    :math:`S_{kv}` is the key/value sequence length.

    Usage:

    >>> import torch
    >>>
    >>> from fairseq2.nn.transformer import CausalAttentionMask
    >>>
    >>> mask = CausalAttentionMask(seq_len=4, key_len=6)
    >>> mask.materialize()
    tensor([[0., -inf, -inf, -inf, -inf, -inf],
            [0.,   0., -inf, -inf, -inf, -inf],
            [0.,   0.,   0., -inf, -inf, -inf],
            [0.,   0.,   0.,   0., -inf, -inf]])
    >>>
    >>> mask = CausalAttentionMask(seq_len=4, key_len=4, attn_window_len=2)
    >>> mask.materialize()
    tensor([[0.,   -inf, -inf, -inf],
            [0.,     0., -inf, -inf],
            [-inf,   0.,   0., -inf],
            [-inf, -inf,   0.,   0.]])
    """

    def __init__(
        self,
        seq_len: int,
        key_len: int,
        *,
        attn_len: Optional[int] = None,
        attn_window_len: Optional[int] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param seq_len:
            The sequence length.
        :param key_len:
            The key/value sequence length.
        :param attn_len:
            The sequence length, starting from the end of the sequence, for
            which to compute the mask.
        :param attn_window_len:
            The attention window length as described in Section 3.1 of
            :cite:t:`https://doi.org/10.48550/arxiv.2004.05150`. If ``None``,
            constructs a full causal attention mask.
        """
        super().__init__()

        self.seq_len = seq_len
        self.key_len = key_len
        self.attn_len = attn_len
        self.attn_window_len = attn_window_len

        self.device, self.dtype = device, dtype

    @finaloverride
    def _do_materialize(self) -> Tensor:
        return _create_causal_attention_mask(
            self.seq_len,
            self.key_len,
            self.attn_len,
            self.attn_window_len,
            self.device,
            self.dtype,
        )


class CausalAttentionMaskFactory:
    """Constructs instances of :class:`CausalAttentionMask`."""

    def __init__(self, *, attn_window_len: Optional[int] = None) -> None:
        """
        :param attn_window_len:
            The attention window length as described in Section 3.1 of
            :cite:t:`https://doi.org/10.48550/arxiv.2004.05150`. If ``None``,
            constructs a full causal attention mask.
        """
        self.attn_window_len = attn_window_len

    def __call__(
        self,
        seqs: Tensor,
        keys: Tensor,
        *,
        training: bool = True,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Optional[CausalAttentionMask]:
        attn_len: Optional[int]

        attn_len = seqs.size(1)

        if training or state_bag is None:
            seq_len = attn_len
        else:
            seq_len = state_bag.step_nr + attn_len

        if seqs is keys:  # Self attention
            key_len = seq_len
        else:
            key_len = keys.size(1)

        if seq_len > key_len:
            raise ValueError(
                f"The sequence length of `seqs` must be less than or equal to the sequence length of `keys` ({key_len}), but is {seq_len} instead."
            )

        if attn_len <= 1:
            # Return `None` if the sequence has a length of 1 during training;
            # or if we attend to past steps during incremental decoding.
            return None

        # PyTorch SDPA does not support `attn_len`; set it to `None` if it is
        # redundant.
        if attn_len == seq_len:
            attn_len = None

        return CausalAttentionMask(
            seq_len,
            key_len,
            attn_len=attn_len,
            attn_window_len=self.attn_window_len,
            device=seqs.device,
            dtype=seqs.dtype,
        )

    def __repr__(self) -> str:
        if self.attn_window_len is None:
            return "CausalAttentionMaskFactory()"

        return f"CausalAttentionMaskFactory(attn_window_len={self.attn_window_len})"


@final
class ALiBiMask(AttentionMask):
    """Represents an ALiBi attention mask as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2108.12409`.

    *Shape:* :math:`(H,S,S_{kv})`, where :math:`H` is the number of attention
    heads, :math:`S` is the sequence length, and :math:`S_{kv}` is the key/value
    sequence length.
    """

    def __init__(
        self,
        seq_len: int,
        key_len: int,
        num_attn_heads: int,
        *,
        attn_len: Optional[int] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param seq_len:
            The sequence length.
        :param key_len:
            The key/value sequence length.
        :param num_attn_heads:
            The number of attention heads.
        :param attn_len:
            The sequence length, starting from the end of the sequence, for
            which to compute the mask.
        """
        super().__init__()

        if num_attn_heads % 2 != 0:
            raise ValueError(
                f"`num_attn_heads` must be even, but is {num_attn_heads} instead."
            )

        self.seq_len = seq_len
        self.key_len = key_len
        self.num_attn_heads = num_attn_heads
        self.attn_len = attn_len

        self.device, self.dtype = device, dtype

    @finaloverride
    def _do_materialize(self) -> Tensor:
        attn_len = self.seq_len if self.attn_len is None else self.attn_len

        # (H)
        powers = torch.arange(1, 1 + self.num_attn_heads, device=self.device)

        # (H)
        slopes = torch.pow(2 ** (-8 / self.num_attn_heads), powers)

        # (S_kv)
        steps = torch.arange(self.key_len, device=self.device)

        # (S_kv) -> (H, S, S_kv)
        steps = steps[None, None, :].expand(self.num_attn_heads, attn_len, -1)

        # (H, S, S_kv) * (H, 1, 1) -> (H, S, S_kv)
        mask = steps * slopes[:, None, None]

        mask = mask.to(self.dtype)

        # If the attention length is 1, avoid constructing the causal mask.
        if attn_len == 1:
            # Ensure that we do not attend to keys beyond the sequence length.
            if (causal := self.key_len - self.seq_len) > 0:
                mask[:, :, -causal:] = -torch.inf
        else:
            # (S, S_kv)
            causal_mask = _create_causal_attention_mask(
                self.seq_len, self.key_len, attn_len, None, self.device, self.dtype
            )

            # (H, S, S_kv) + (S, S_kv) -> (H, S, S_kv)
            mask = mask + causal_mask

        return mask


class ALiBiMaskFactory:
    """Constructs instances of :class:`ALiBiMask`."""

    def __init__(self, num_attn_heads: int) -> None:
        """
        :param num_attn_heads:
            The number of attention heads.
        """
        self.num_attn_heads = num_attn_heads

    def __call__(
        self,
        seqs: Tensor,
        keys: Tensor,
        *,
        training: bool = True,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Optional[ALiBiMask]:
        attn_len: Optional[int]

        attn_len = seqs.size(1)

        if training or state_bag is None:
            seq_len = attn_len
        else:
            seq_len = state_bag.step_nr + attn_len

        if seqs is keys:  # Self attention
            key_len = seq_len
        else:
            key_len = keys.size(1)

        if seq_len > key_len:
            raise ValueError(
                f"The sequence length of `seqs` must be less than or equal to the sequence length of `keys` ({key_len}), but is {seq_len} instead."
            )

        if attn_len == seq_len:
            attn_len = None

        return ALiBiMask(
            seq_len,
            key_len,
            self.num_attn_heads,
            attn_len=attn_len,
            device=seqs.device,
            dtype=seqs.dtype,
        )

    def __repr__(self) -> str:
        return f"ALiBiMaskFactory(num_attn_heads={self.num_attn_heads})"


def _create_causal_attention_mask(
    seq_len: int,
    key_len: int,
    attn_len: Optional[int],
    attn_window_len: Optional[int],
    device: Optional[Device],
    dtype: Optional[DataType],
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    # As of PyTorch 2.0, `triu` does not support bf16.
    dt = torch.float32 if dtype == torch.bfloat16 else dtype

    # (S, S_kv)
    mask = torch.ones((seq_len, key_len), device=device, dtype=dt)

    mask.tril_(diagonal=0)

    if attn_window_len is not None:
        mask.triu_(diagonal=1 - attn_window_len)

    if attn_len is not None and attn_len != seq_len:
        mask = mask[-attn_len:]

    mask.log_()

    return mask.to(dtype)
