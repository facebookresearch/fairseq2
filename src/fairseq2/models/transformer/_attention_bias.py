# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar, final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn import BatchLayout


class AttentionBias(ABC):
    @final
    def materialize(
        self,
        q_layout: BatchLayout,
        k_layout: BatchLayout,
        device: Device,
        dtype: DataType,
    ) -> Tensor:
        if q_layout.packed ^ k_layout.packed:
            raise ValueError("`q_layout` and `k_layout` must be both packed.")

        num_q_seqs = len(q_layout.seq_lens)
        num_k_seqs = len(k_layout.seq_lens)

        if num_q_seqs != num_k_seqs:
            raise ValueError(
                f"`len(q_layout.seq_lens)` and `len(k_layout.seq_lens)` must be equal, but they are {num_q_seqs} and {num_k_seqs} instead."
            )

        if num_q_seqs == 0:
            return torch.zeros((), device=device, dtype=dtype)

        q_len = q_layout.width
        k_len = k_layout.width

        if q_layout.packed:
            return self._create_block_bias_tensor(
                q_layout.seq_lens, q_len, k_layout.seq_lens, k_len, device, dtype
            )

        if not q_layout.padded and not k_layout.padded:
            return self._create_bias_tensor(q_len, k_len, device, dtype)

        biases = []

        for idx in range(num_q_seqs):
            q_seq_lens = [q_layout.seq_lens[idx]]
            k_seq_lens = [k_layout.seq_lens[idx]]

            bias = self._create_block_bias_tensor(
                q_seq_lens, q_len, k_seq_lens, k_len, device, dtype
            )

            if bias.ndim == 2:
                bias = bias.unsqueeze(0)  # add head dim

            biases.append(bias)

        return torch.stack(biases)

    @final
    def _create_block_bias_tensor(
        self,
        q_seq_lens: Sequence[int],
        q_len: int,
        k_seq_lens: Sequence[int],
        k_len: int,
        device: Device,
        dtype: DataType,
    ) -> Tensor:
        q_seq_ranges = self._get_seq_ranges(q_seq_lens)
        k_seq_ranges = self._get_seq_ranges(k_seq_lens)

        if len(q_seq_ranges) == 1:
            q_begin, q_end = q_seq_ranges[0]
            k_begin, k_end = k_seq_ranges[0]

            if q_begin == 0 and q_end == q_len and k_begin == 0 and k_end == k_len:
                return self._create_bias_tensor(q_len, k_len, device, dtype)

        bias = torch.full((q_len, k_len), -torch.inf, device=device, dtype=dtype)

        q_seq_ranges.append((q_len, q_len))
        k_seq_ranges.append((k_len, k_len))

        prev_q_end = 0
        prev_k_end = 0

        for (q_begin, q_end), (k_begin, k_end) in zip(q_seq_ranges, k_seq_ranges):
            if prev_q_end != q_begin and prev_k_end != k_begin:  # pad
                bias[prev_q_end:q_begin, prev_k_end:k_begin] = 0.0

            if q_begin != q_end and k_begin != k_end:
                bias[q_begin:q_end, k_begin:k_end] = self._create_bias_tensor(
                    q_end - q_begin, k_end - k_begin, device, dtype
                )

            prev_q_end = q_end
            prev_k_end = k_end

        return bias

    @staticmethod
    def _get_seq_ranges(seq_lens: Sequence[int]) -> list[tuple[int, int]]:
        seq_ranges = []

        seq_beg = 0

        for seq_len in seq_lens:
            seq_end = seq_beg + seq_len

            seq_ranges.append((seq_beg, seq_end))

            seq_beg = seq_end

        return seq_ranges

    @abstractmethod
    def _create_bias_tensor(
        self, q_len: int, k_len: int, device: Device, dtype: DataType
    ) -> Tensor: ...


@final
class IdentityBias(AttentionBias):
    def __eq__(self, other: object) -> bool:
        if isinstance(other, IdentityBias):
            return True

        return NotImplemented

    def __hash__(self) -> int:
        return hash("identity")

    @override
    def _create_bias_tensor(
        self, q_len: int, k_len: int, device: Device, dtype: DataType
    ) -> Tensor:
        # (S, S_kv)
        return torch.zeros((q_len, k_len), device=device, dtype=dtype)

    def __repr__(self) -> str:
        return "IdentityBias()"


@final
class CausalAttentionBias(AttentionBias):
    """Represents a causal attention bias."""

    attn_window_len: int | None

    def __init__(self, *, attn_window_len: int | None = None) -> None:
        """
        :param attn_window_len: The attention window length as described in
            Section 3.1 of :cite:t:`https://doi.org/10.48550/arxiv.2004.05150`.
            If ``None``, constructs a full causal attention.
        """
        super().__init__()

        self.attn_window_len = attn_window_len

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CausalAttentionBias):
            return self.attn_window_len == other.attn_window_len

        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.attn_window_len,))

    @override
    def _create_bias_tensor(
        self, q_len: int, k_len: int, device: Device, dtype: DataType
    ) -> Tensor:
        if self.attn_window_len is None:
            attn_window_len = q_len
        else:
            attn_window_len = self.attn_window_len

        # (S, S_kv)
        return _create_causal_bias_tensor(
            q_len,
            k_len,
            attn_window_len,
            from_bottom_right=True,
            device=device,
            dtype=dtype,
        )

    def __repr__(self) -> str:
        if self.attn_window_len is None:
            return "CausalAttentionBias()"

        return f"CausalAttentionBias(attn_window_len={self.attn_window_len})"


@final
class ALiBiAttentionBias(AttentionBias):
    """
    Represents an ALiBi attention bias as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2108.12409`.
    """

    num_heads: int

    def __init__(self, num_heads: int) -> None:
        self.num_heads = num_heads

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ALiBiAttentionBias):
            return self.num_heads == other.num_heads

        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.num_heads,))

    @override
    def _create_bias_tensor(
        self, q_len: int, k_len: int, device: Device, dtype: DataType
    ) -> Tensor:
        num_heads = self.num_heads

        # (H)
        exponents = torch.arange(1, 1 + num_heads, device=device)

        # (H)
        slopes = torch.pow(2 ** (-8 / num_heads), exponents)

        # (S_kv)
        steps = torch.arange(k_len, device=device)

        # (S_kv) -> (H, S, S_kv)
        steps = steps[None, None, :].expand(num_heads, q_len, -1)

        # (H, S, S_kv) * (H, 1, 1) -> (H, S, S_kv)
        bias = steps * slopes[:, None, None]

        bias = bias.to(dtype)

        if q_len > 1:
            # (S, S_kv)
            causal_bias = _create_causal_bias_tensor(
                q_len,
                k_len,
                q_len,
                from_bottom_right=True,
                device=device,
                dtype=dtype,
            )

            # (H, S, S_kv) + (S, S_kv) -> (H, S, S_kv)
            bias = bias + causal_bias

        # (H, S, S_kv)
        return bias

    def __repr__(self) -> str:
        return f"ALiBiAttentionBias(num_heads={self.num_heads})"


def _create_causal_bias_tensor(
    q_len: int,
    k_len: int,
    attn_window_len: int,
    from_bottom_right: bool,
    device: Device,
    dtype: DataType,
) -> Tensor:
    # (S, S_kv)
    bias = torch.ones((q_len, k_len), device=device, dtype=dtype)

    if from_bottom_right:
        d = k_len - q_len
    else:
        d = 0

    bias.tril_(diagonal=d)

    if attn_window_len < q_len:
        bias.triu_(diagonal=d - attn_window_len + 1)

    bias.log_()

    return bias


T = TypeVar("T")


@final
class AttentionBiasCache:
    _data: dict[tuple[AttentionBias, str], object]

    def __init__(self) -> None:
        self._data = {}

    def maybe_get(self, bias: AttentionBias, impl: str, kls: type[T]) -> T | None:
        value = self._data.get((bias, impl))
        if value is None:
            return None

        if not isinstance(value, kls):
            raise TypeError(
                f"The attention bias value is expected to be of type `{kls}`, but is of type `{type(bias)}` instead."
            )

        return value

    def set(self, bias: AttentionBias, impl: str, value: object) -> None:
        self._data[(bias, impl)] = value

    def clear(self) -> None:
        self._data.clear()
