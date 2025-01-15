# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, final

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import dropout, scaled_dot_product_attention, softmax
from typing_extensions import override

from fairseq2.logging import log
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer._attention_mask import AttentionMask, CausalAttentionMask


class SDPA(Module, ABC):
    """Computes scaled dot-product attention."""

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        keys: Tensor,
        key_padding_mask: PaddingMask | None,
        values: Tensor,
        *,
        attn_mask: AttentionMask | None = None,
        needs_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        :param seqs:
            The sequences to query. *Shape:* :math:`(N,H,S,K)`, where :math:`N`
            is the batch size, :math:`H` is the number of heads, :math:`S` is
            the sequence length, and :math:`K` is the key size.
        :param keys:
            The keys. *Shape:* :math:`(N,H,S_{kv},K)`, where :math:`N` is the
            batch size, :math:`H` is the number of heads, :math:`S_{kv}` is the
            key/value sequence length, and :math:`K` is the key size.
        :param key_padding_mask:
            The padding mask indicating which key positions to ignore for the
            purpose of attention. *Shape:* :math:`(N,S_{kv})`, where :math:`N`
            is the batch size and :math:`S_{kv}` is the key/value sequence
            length.
        :param values:
            The values. *Shape:* :math:`(N,H,S_{kv},V)`, where :math:`N` is the
            batch size, :math:`H` is the number of heads, :math:`S_{kv}` is the
            key/value sequence length, and :math:`V` is the value size.
        :param attn_mask:
            The mask that will be added to attention weights before computing
            the attention. *Shape:* :math:`([H],S,S_{kv})`, where :math:`H` is
            the number of heads, :math:`S` is the sequence length, and
            :math:`S_{kv}` is the key/value sequence length.
        :param needs_weights:
            If ``True``, returns the attention weights.

        :returns:
            - The attention values. *Shape:* :math:`(N,H,S,V)`, where :math:`N`
              is the batch size, :math:`H` is the number of heads, :math:`S` is
              the sequence length, and :math:`V` is the value size.
            - The attention weights. *Shape:* :math:`(N,H,S,S_{kv})`, where
              :math:`N` is the batch size, :math:`H` is the number of heads,
              :math:`S` is the sequence length, and :math:`S_{kv}` is the
              key/value sequence length.
        """


@final
class TorchSDPA(SDPA):
    """Computes scaled dot-product attention using PyTorch SDPA v2."""

    attn_dropout_p: float

    _has_warned: bool
    _enable_memory_efficient: bool

    def __init__(self, *, attn_dropout_p: float = 0.0) -> None:
        """
        :param attn_dropout_p:
            The dropout probability on attention weights.
        """
        super().__init__()

        self.attn_dropout_p = attn_dropout_p

        self._has_warned = False
        self._enable_memory_efficient = True

    def enable_memory_efficient(self, value: bool = True) -> None:
        """Enable or disable the memory efficient SDPA implementation."""
        self._enable_memory_efficient = value

    @override
    def forward(
        self,
        seqs: Tensor,
        keys: Tensor,
        key_padding_mask: PaddingMask | None,
        values: Tensor,
        *,
        attn_mask: AttentionMask | None = None,
        needs_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if needs_weights:
            if not self._has_warned:
                log.warning("`TorchSDPA` has to fall back to the naive SDPA implementation because of `needs_weights` set to `True`.")  # fmt: skip

                self._has_warned = True

            return _naive_scaled_dot_product_attention(
                seqs,
                keys,
                key_padding_mask,
                values,
                attn_mask,
                self.attn_dropout_p,
                needs_weights,
                self.training,
            )

        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.attn_dropout_p

        is_causal = False

        if key_padding_mask is not None:
            mask = key_padding_mask.materialize_as(seqs)

            # (N, S_kv) -> (N, 1, 1, S_kv)
            mask = mask[:, None, None, :]

            # (N, 1, 1, S_kv) -> (N, H, 1, S_kv)
            mask = mask.expand(-1, seqs.size(1), -1, -1)

            if attn_mask is not None:
                # (N, H, 1, S_kv) + ([N, H], S, S_kv) -> (N, H, S, S_kv)
                mask = mask + attn_mask.materialize()
        elif isinstance(attn_mask, CausalAttentionMask):
            # PyTorch SDPA supports only full causal attention.
            if attn_mask.full_attention():
                mask = None

                is_causal = True
            else:
                # ([N, H], S, S_kv)
                mask = attn_mask.materialize()
        elif attn_mask is not None:
            # ([N, H], S, S_kv)
            mask = attn_mask.materialize()
        else:
            mask = None

        with _with_memory_efficient_kernel(self._enable_memory_efficient):
            attn = scaled_dot_product_attention(
                seqs,
                keys,
                values,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

        return attn, None

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"attn_dropout_p={self.attn_dropout_p:G}"


def enable_memory_efficient_torch_sdpa(module: Module, value: bool) -> None:
    """Enable or disable the memory efficient PyTorch SDPA implementation."""
    for m in module.modules():
        if isinstance(m, TorchSDPA):
            m.enable_memory_efficient(value)


try:
    from torch.backends.cuda import enable_mem_efficient_sdp, mem_efficient_sdp_enabled

    @contextmanager
    def _with_memory_efficient_kernel(value: bool) -> Iterator[None]:
        original_value = mem_efficient_sdp_enabled()

        enable_mem_efficient_sdp(value)

        try:
            yield
        finally:
            enable_mem_efficient_sdp(original_value)

except ImportError:

    @contextmanager
    def _with_memory_efficient_kernel(value: bool) -> Iterator[None]:
        yield


@final
class NaiveSDPA(SDPA):
    """Computes scaled dot-product attention using a Python implementation."""

    attn_dropout_p: float

    def __init__(self, *, attn_dropout_p: float = 0.0) -> None:
        """
        :param attn_dropout_p:
            The dropout probability on attention weights.
        """
        super().__init__()

        self.attn_dropout_p = attn_dropout_p

    @override
    def forward(
        self,
        seqs: Tensor,
        keys: Tensor,
        key_padding_mask: PaddingMask | None,
        values: Tensor,
        *,
        attn_mask: AttentionMask | None = None,
        needs_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        return _naive_scaled_dot_product_attention(
            seqs,
            keys,
            key_padding_mask,
            values,
            attn_mask,
            self.attn_dropout_p,
            needs_weights,
            self.training,
        )

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"attn_dropout_p={self.attn_dropout_p:G}"


def _naive_scaled_dot_product_attention(
    seqs: Tensor,
    keys: Tensor,
    key_padding_mask: PaddingMask | None,
    values: Tensor,
    attn_mask: AttentionMask | None,
    dropout_p: float,
    needs_weights: bool,
    training: bool,
) -> tuple[Tensor, Tensor | None]:
    # (N, H, S, K) @ (N, H, K, S_kv) = (N, H, S, S_kv)
    attn_weights = torch.matmul(seqs, keys.transpose(-1, -2))

    attn_weights = attn_weights * (seqs.size(-1) ** -0.5)

    if attn_mask is not None:
        # ([N, H], S, S_kv)
        m = attn_mask.materialize()

        # (N, H, S, S_kv) + ([N, H], S, S_kv) -> (N, H, S, S_kv)
        attn_weights = attn_weights + m

    if key_padding_mask is not None:
        # (N, S_kv)
        m = key_padding_mask.materialize()

        m = m[:, None, None, :]

        # (N, H, S, S_kv) + (N, 1, 1, S_kv) -> (N. H, S, S_kv)
        attn_weights = torch.where(m, attn_weights, -torch.inf)

    # For numerical stability run in single precision.
    attn_weights = softmax(attn_weights, dim=-1, dtype=torch.float32)

    attn_weights = attn_weights.type_as(seqs)

    if training and dropout_p > 0.0:
        attn_weights = dropout(attn_weights, dropout_p)

    # (N, H, S, S_kv) @ (N, H, S_kv, V) = (N, H, S, V)
    attn = torch.matmul(attn_weights, values)

    return attn, attn_weights if needs_weights else None


class SDPAFactory(Protocol):
    """Constructs instances of :class:`SDPA`."""

    def __call__(self, *, attn_dropout_p: float = 0.0) -> SDPA:
        """
        :param attn_dropout_p:
            The dropout probability on attention weights.
        """


def _get_fallback_sdpa_factory() -> SDPAFactory:
    return TorchSDPA


_sdpa_factory: SDPAFactory = _get_fallback_sdpa_factory()


def set_default_sdpa_factory(factory: SDPAFactory | None) -> None:
    """Set the default :class:`SDPA` factory."""
    global _sdpa_factory

    if factory is not None:
        _sdpa_factory = factory
    else:
        _sdpa_factory = _get_fallback_sdpa_factory()


def create_default_sdpa(*, attn_dropout_p: float = 0.0) -> SDPA:
    """Make an instance of the default :class:`SDPA`.

    :param attn_dropout_p:
        The dropout probability on attention weights.
    """
    return _sdpa_factory(attn_dropout_p=attn_dropout_p)


@contextmanager
def default_sdpa_factory(factory: SDPAFactory | None) -> Iterator[None]:
    """Set a temporary default :class:`SDPA` factory."""
    original_factory = _sdpa_factory

    set_default_sdpa_factory(factory)

    try:
        yield
    finally:
        set_default_sdpa_factory(original_factory)
