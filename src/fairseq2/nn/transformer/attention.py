# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator, Optional, Protocol, Tuple, final

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import dropout, softmax

from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer.attention_mask import AttentionMask, CausalAttentionMask
from fairseq2.typing import finaloverride

logger = logging.getLogger(__name__)


class SDPA(Module, ABC):
    """Computes scaled dot-product attention."""

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        keys: Tensor,
        key_padding_mask: Optional[PaddingMask],
        values: Tensor,
        *,
        attn_mask: Optional[AttentionMask] = None,
        needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
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

    def __init__(self, *, attn_dropout_p: float = 0.0) -> None:
        """
        :param attn_dropout_p:
            The dropout probability on attention weights.
        """
        super().__init__()

        self._has_warned = False

        self.attn_dropout_p = attn_dropout_p

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        keys: Tensor,
        key_padding_mask: Optional[PaddingMask],
        values: Tensor,
        *,
        attn_mask: Optional[AttentionMask] = None,
        needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if needs_weights:
            if not self._has_warned:
                logger.warning("`TorchSDPA` has to fall back to the naive SDPA implementation because of `needs_weights` set to `True`.")  # fmt: skip

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
            mask = key_padding_mask.materialize()

            # (N, S_kv) -> (N, 1, 1, S_kv)
            mask = mask[:, None, None, :]

            # (N, 1, 1, S_kv) -> (N, H, S, S_kv)
            mask = mask.expand(-1, seqs.size(1), seqs.size(2), -1)

            if attn_mask is not None:
                # ([H], S, S_kv)
                m = attn_mask.materialize()

                # (N, H, S, S_kv)
                mask = torch.where(mask, m, -torch.inf)
        elif isinstance(attn_mask, CausalAttentionMask):
            # PyTorch SDPA supports only full causal attention.
            if attn_mask.attn_len is None and attn_mask.attn_window_len is None:
                mask = None

                is_causal = True
            else:
                # ([H], S, S_kv)
                mask = attn_mask.materialize()
        elif attn_mask is not None:
            # ([H], S, S_kv)
            mask = attn_mask.materialize()
        else:
            mask = None

        attn = F.scaled_dot_product_attention(  # type: ignore[attr-defined]
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
        return f"attn_dropout_p={self.attn_dropout_p}"


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

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        keys: Tensor,
        key_padding_mask: Optional[PaddingMask],
        values: Tensor,
        *,
        attn_mask: Optional[AttentionMask] = None,
        needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
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
        return f"attn_dropout_p={self.attn_dropout_p}"


def _naive_scaled_dot_product_attention(
    seqs: Tensor,
    keys: Tensor,
    key_padding_mask: Optional[PaddingMask],
    values: Tensor,
    attn_mask: Optional[AttentionMask],
    dropout_p: float,
    needs_weights: bool,
    training: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    # (N, H, S, K) @ (N, H, K, S_kv) = (N, H, S, S_kv)
    attn_weights = torch.matmul(seqs, keys.transpose(-1, -2))

    attn_weights = attn_weights * (seqs.size(-1) ** -0.5)

    if attn_mask is not None:
        # (S, S_kv)
        m = attn_mask.materialize()

        # (N, H, S, S_kv) + (S, S_kv) -> (N, H, S, S_kv)
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


def set_default_sdpa_factory(factory: Optional[SDPAFactory]) -> None:
    """Set the default :class:`SDPA` factory."""
    global _sdpa_factory

    if factory is not None:
        _sdpa_factory = factory
    else:
        _sdpa_factory = _get_fallback_sdpa_factory()


def create_default_sdpa(*, attn_dropout_p: float = 0.0) -> SDPA:
    """Create an instance of the default :class:`SDPA`.

    :param attn_dropout_p:
        The dropout probability on attention weights.
    """
    return _sdpa_factory(attn_dropout_p=attn_dropout_p)


@contextmanager
def default_sdpa_factory(factory: Optional[SDPAFactory]) -> Generator[None, None, None]:
    """Set a temporary default :class:`SDPA` factory."""
    original_factory = _sdpa_factory

    set_default_sdpa_factory(factory)

    try:
        yield
    finally:
        set_default_sdpa_factory(original_factory)
