# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.nn import IncrementalStateBag
from fairseq2.typing import DataType, Device


class AttentionMask(ABC):
    """Represents an attention mask."""

    @abstractmethod
    def materialize(self) -> Tensor:
        """Materialize the attention mask tensor."""


class AbstractAttentionMask(AttentionMask):
    """Provides a skeletal implementation of :class:`AttentionMask`."""

    _materialized: Tensor | None

    def __init__(self) -> None:
        self._materialized = None

    @final
    @override
    def materialize(self) -> Tensor:
        if self._materialized is None:
            self._materialized = self._do_materialize()

        return self._materialized

    @abstractmethod
    def _do_materialize(self) -> Tensor: ...


class AttentionMaskFactory(Protocol):
    """Constructs instances of :class:`AttentionMask`."""

    def __call__(
        self,
        seqs: Tensor,
        keys: Tensor,
        *,
        training: bool = True,
        state_bag: IncrementalStateBag | None = None,
    ) -> AttentionMask | None:
        """
        :param seqs:
            The sequences for which to make a mask. *Shape:* :math:`(N,S,M)`,
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

    _mask: Tensor

    def __init__(self, mask: Tensor) -> None:
        """
        :param mask:
            The custom attention mask tensor.
        """
        self._mask = mask

    @override
    def materialize(self) -> Tensor:
        return self._mask


@final
class CausalAttentionMask(AbstractAttentionMask):
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

    _seq_len: int
    _key_len: int
    _attn_len: int | None
    _attn_window_len: int | None
    _device: Device | None
    _dtype: DataType | None

    def __init__(
        self,
        seq_len: int,
        key_len: int,
        *,
        attn_len: int | None = None,
        attn_window_len: int | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
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

        self._seq_len = seq_len
        self._key_len = key_len
        self._attn_len = attn_len
        self._attn_window_len = attn_window_len

        self._device, self._dtype = device, dtype

    @override
    def _do_materialize(self) -> Tensor:
        return _create_causal_attention_mask(
            self._seq_len,
            self._key_len,
            self._attn_len,
            self._attn_window_len,
            self._device,
            self._dtype,
        )

    def full_attention(self) -> bool:
        """Return ``True`` if this is a full causal attention mask."""
        return self._attn_len is None and self._attn_window_len is None


@final
class CausalAttentionMaskFactory(AttentionMaskFactory):
    """Constructs instances of :class:`CausalAttentionMask`."""

    _attn_window_len: int | None

    def __init__(self, *, attn_window_len: int | None = None) -> None:
        """
        :param attn_window_len:
            The attention window length as described in Section 3.1 of
            :cite:t:`https://doi.org/10.48550/arxiv.2004.05150`. If ``None``,
            constructs a full causal attention mask.
        """
        self._attn_window_len = attn_window_len

    def __call__(
        self,
        seqs: Tensor,
        keys: Tensor,
        *,
        training: bool = True,
        state_bag: IncrementalStateBag | None = None,
    ) -> CausalAttentionMask | None:
        attn_len: int | None

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
            attn_window_len=self._attn_window_len,
            device=seqs.device,
            dtype=seqs.dtype,
        )

    def __repr__(self) -> str:
        if self._attn_window_len is None:
            return "CausalAttentionMaskFactory()"

        return f"CausalAttentionMaskFactory(attn_window_len={self._attn_window_len})"


@final
class ALiBiMask(AbstractAttentionMask):
    """Represents an ALiBi attention mask as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2108.12409`.

    *Shape:* :math:`(H,S,S_{kv})`, where :math:`H` is the number of attention
    heads, :math:`S` is the sequence length, and :math:`S_{kv}` is the key/value
    sequence length.
    """

    _seq_len: int
    _key_len: int
    _num_attn_heads: int
    _attn_len: int | None = None
    _device: Device | None = None
    _dtype: DataType | None = None

    def __init__(
        self,
        seq_len: int,
        key_len: int,
        num_attn_heads: int,
        *,
        attn_len: int | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
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

        self._seq_len = seq_len
        self._key_len = key_len
        self._num_attn_heads = num_attn_heads
        self._attn_len = attn_len

        self._device, self._dtype = device, dtype

    @override
    def _do_materialize(self) -> Tensor:
        attn_len = self._seq_len if self._attn_len is None else self._attn_len

        # (H)
        powers = torch.arange(1, 1 + self._num_attn_heads, device=self._device)

        # (H)
        slopes = torch.pow(2 ** (-8 / self._num_attn_heads), powers)

        # (S_kv)
        steps = torch.arange(self._key_len, device=self._device)

        # (S_kv) -> (H, S, S_kv)
        steps = steps[None, None, :].expand(self._num_attn_heads, attn_len, -1)

        # (H, S, S_kv) * (H, 1, 1) -> (H, S, S_kv)
        mask = steps * slopes[:, None, None]

        mask = mask.to(self._dtype)

        # If the attention length is 1, avoid constructing the causal mask.
        if attn_len == 1:
            # Ensure that we do not attend to keys beyond the sequence length.
            if (causal := self._key_len - self._seq_len) > 0:
                mask[:, :, -causal:] = -torch.inf
        else:
            # (S, S_kv)
            causal_mask = _create_causal_attention_mask(
                self._seq_len, self._key_len, attn_len, None, self._device, self._dtype
            )

            # (H, S, S_kv) + (S, S_kv) -> (H, S, S_kv)
            mask = mask + causal_mask

        return mask


@final
class ALiBiMaskFactory(AttentionMaskFactory):
    """Constructs instances of :class:`ALiBiMask`."""

    _num_attn_heads: int

    def __init__(self, num_attn_heads: int) -> None:
        """
        :param num_attn_heads:
            The number of attention heads.
        """
        self._num_attn_heads = num_attn_heads

    def __call__(
        self,
        seqs: Tensor,
        keys: Tensor,
        *,
        training: bool = True,
        state_bag: IncrementalStateBag | None = None,
    ) -> ALiBiMask | None:
        attn_len: int | None

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
            self._num_attn_heads,
            attn_len=attn_len,
            device=seqs.device,
            dtype=seqs.dtype,
        )

    def __repr__(self) -> str:
        return f"ALiBiMaskFactory(num_attn_heads={self._num_attn_heads})"


def _create_causal_attention_mask(
    seq_len: int,
    key_len: int,
    attn_len: int | None,
    attn_window_len: int | None,
    device: Device | None,
    dtype: DataType | None,
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
