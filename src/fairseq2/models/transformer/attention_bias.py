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
    @abstractmethod
    def create_bias_tensor(
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
    def create_bias_tensor(
        self, q_len: int, k_len: int, device: Device, dtype: DataType
    ) -> Tensor:
        # (S, S_kv)
        return torch.zeros((q_len, k_len), device=device, dtype=dtype)

    def __repr__(self) -> str:
        return "IdentityBias()"


@final
class CausalAttentionBias(AttentionBias):
    """Represents a causal attention bias."""

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
    def create_bias_tensor(
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
class ChunkedAttentionBias(AttentionBias):
    """Represents a chunked attention bias."""

    def __init__(self, attn_chunk_size: int) -> None:
        """
        :param attn_chunk_size:
            The attention chunk size.
        """
        self.attn_chunk_size = attn_chunk_size

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ChunkedAttentionBias):
            return self.attn_chunk_size == other.attn_chunk_size

        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.attn_chunk_size,))

    @override
    def create_bias_tensor(
        self, q_len: int, k_len: int, device: Device, dtype: DataType
    ) -> Tensor:
        if q_len != k_len:
            raise ValueError(f"`q_len` and `k_len` must be equal: {q_len} != {k_len}")

        # (S, S)
        block_pos = (
            torch.arange(q_len, device=device).unsqueeze(0) // self.attn_chunk_size
        ) - (torch.arange(q_len, device=device).unsqueeze(1) // self.attn_chunk_size)
        token_pos = torch.arange(q_len, device=device).unsqueeze(0) - torch.arange(
            q_len, device=device
        ).unsqueeze(1)

        mask: Tensor = (block_pos == 0) & (token_pos <= 0)

        mask = mask.to(dtype)

        # (S, S)
        return mask

    def __repr__(self) -> str:
        return f"ChunkedAttentionBias(attn_chunk_size={self.attn_chunk_size})"


@final
class ALiBiAttentionBias(AttentionBias):
    """
    Represents an ALiBi attention bias as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2108.12409`.
    """

    def __init__(self, num_heads: int) -> None:
        self.num_heads = num_heads

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ALiBiAttentionBias):
            return self.num_heads == other.num_heads

        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.num_heads,))

    @override
    def create_bias_tensor(
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


def materialize_attention_bias(
    bias: AttentionBias,
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
        return _create_block_bias_tensor(
            bias, q_layout.seq_lens, q_len, k_layout.seq_lens, k_len, device, dtype
        )

    if not q_layout.padded and not k_layout.padded:
        return bias.create_bias_tensor(q_len, k_len, device, dtype)

    tensors = []

    for idx in range(num_q_seqs):
        q_seq_lens = [q_layout.seq_lens[idx]]
        k_seq_lens = [k_layout.seq_lens[idx]]

        tensor = _create_block_bias_tensor(
            bias, q_seq_lens, q_len, k_seq_lens, k_len, device, dtype
        )

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)  # add head dim

        tensors.append(tensor)

    return torch.stack(tensors)


def _create_block_bias_tensor(
    bias: AttentionBias,
    q_seq_lens: Sequence[int],
    q_len: int,
    k_seq_lens: Sequence[int],
    k_len: int,
    device: Device,
    dtype: DataType,
) -> Tensor:
    q_seq_ranges = _get_seq_ranges(q_seq_lens)
    k_seq_ranges = _get_seq_ranges(k_seq_lens)

    if len(q_seq_ranges) == 1:
        q_begin, q_end = q_seq_ranges[0]
        k_begin, k_end = k_seq_ranges[0]

        if q_begin == 0 and q_end == q_len and k_begin == 0 and k_end == k_len:
            return bias.create_bias_tensor(q_len, k_len, device, dtype)

    tensor = torch.full((q_len, k_len), -torch.inf, device=device, dtype=dtype)

    q_seq_ranges.append((q_len, q_len))
    k_seq_ranges.append((k_len, k_len))

    prev_q_end = 0
    prev_k_end = 0

    for (q_begin, q_end), (k_begin, k_end) in zip(q_seq_ranges, k_seq_ranges):
        if prev_q_end != q_begin and prev_k_end != k_begin:  # pad
            tensor[prev_q_end:q_begin, prev_k_end:k_begin] = 0.0

        if q_begin != q_end and k_begin != k_end:
            tensor[q_begin:q_end, k_begin:k_end] = bias.create_bias_tensor(
                q_end - q_begin, k_end - k_begin, device, dtype
            )

        prev_q_end = q_end
        prev_k_end = k_end

    return tensor


def _get_seq_ranges(seq_lens: Sequence[int]) -> list[tuple[int, int]]:
    seq_ranges = []

    seq_beg = 0

    for seq_len in seq_lens:
        seq_end = seq_beg + seq_len

        seq_ranges.append((seq_beg, seq_end))

        seq_beg = seq_end

    return seq_ranges


T = TypeVar("T")


@final
class AttentionBiasCache:
    def __init__(self) -> None:
        self._data: dict[tuple[AttentionBias, str], object] = {}

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


def maybe_get_attention_bias_tensor(
    bias: AttentionBias,
    q: Tensor,
    q_layout: BatchLayout,
    k_layout: BatchLayout,
    bias_cache: AttentionBiasCache,
) -> Tensor | None:
    if isinstance(bias, IdentityBias):
        full_q = not q_layout.packed and not q_layout.padded
        full_k = not k_layout.packed and not k_layout.padded

        if full_q and full_k:
            return None

    if isinstance(bias, CausalAttentionBias) or isinstance(bias, ChunkedAttentionBias):
        if not q_layout.packed:
            if q_layout.max_seq_len == 1:
                return None

    return _get_attention_bias_tensor(  # type: ignore[no-any-return]
        bias, q, q_layout, k_layout, bias_cache
    )


@torch.compiler.disable
def _get_attention_bias_tensor(
    bias: AttentionBias,
    q: Tensor,
    q_layout: BatchLayout,
    k_layout: BatchLayout,
    bias_cache: AttentionBiasCache,
) -> Tensor:
    impl = "tensor"

    tensor = bias_cache.maybe_get(bias, impl, kls=Tensor)
    if tensor is None:
        tensor = materialize_attention_bias(bias, q_layout, k_layout, q.device, q.dtype)

        bias_cache.set(bias, impl, tensor)

    return tensor
