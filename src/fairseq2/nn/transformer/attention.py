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

from fairseq2.typing import finaloverride
from fairseq2.utils.version import is_pt2_or_greater

logger = logging.getLogger(__name__)


class SDPA(Module, ABC):
    """Computes scaled dot-product attention."""

    attn_dropout_p: float

    def __init__(self, *, attn_dropout_p: float = 0.0) -> None:
        """
        :param attn_dropout_p:
            The dropout probability on attention weights.
        """
        super().__init__()

        self.attn_dropout_p = attn_dropout_p

    @abstractmethod
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        *,
        mask: Optional[Tensor] = None,
        needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param queries:
            The queries. *Shape:* :math:`(*,S,K)`, where :math:`*` is any number
            of batch dimensions including none, :math:`S` is the sequence
            length, and :math:`K` is the key size.
        :param keys:
            The keys. *Shape:* :math:`(N,S_{kv},K)`, where :math:`*` is any
            number of batch dimensions including none, :math:`S_{kv}` is the
            key/value sequence length, and :math:`K` is the key size.
        :param values:
            The values. *Shape:* :math:`(N,S_{kv},V)`, where :math:`*` is any
            number of batch dimensions including none, :math:`S_{kv}` is the
            key/value sequence length, and :math:`V` is the value size.
        :param mask:
            The float mask that will be added to the attention weights before
            computing the attention. *Shape:* :math:`(S,S_{kv})` or
            :math:`(*,S,S_{kv})`, where :math:`*` is any number of batch
            dimensions including none, :math:`S` is the sequence length, and
            :math:`S_{kv}` is the key/value sequence length.
        :param needs_weights:
            If ``True``, returns the attention weights.

        :returns:
            - The attention values. *Shape:* :math:`(*,S,V)`, where :math:`*`
              is the same batch dimensions as input, :math:`S` is the sequence
              length, and :math:`V` is the value size.
            - The attention weights. *Shape:* :math:`(*,S,S_{kv})`, where
              :math:`*` is the same batch dimensions as input, :math:`S` is the
              sequence length, and :math:`S_{kv}` is the key/value sequence
              length.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"attn_dropout_p={self.attn_dropout_p}"


@final
class TorchSDPA(SDPA):
    """Computes scaled dot-product attention using PyTorch SDPA v2."""

    def __init__(self, *, attn_dropout_p: float = 0.0) -> None:
        super().__init__(attn_dropout_p=attn_dropout_p)

        if not is_pt2_or_greater():
            raise ValueError("`TorchSDPA` requires PyTorch 2.0.0 or greater.")

        self._has_warned = False

    @finaloverride
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        *,
        mask: Optional[Tensor] = None,
        needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not queries.is_cuda:
            return _naive_scaled_dot_product_attention(
                queries,
                keys,
                values,
                mask,
                self.attn_dropout_p,
                needs_weights,
                self.training,
            )

        if needs_weights:
            if not self._has_warned:
                logger.warning(
                    "`TorchSDPA` has to fall back to the naive SDPA implementation because of `needs_weights` set to `True`."
                )

                self._has_warned = True

            return _naive_scaled_dot_product_attention(
                queries,
                keys,
                values,
                mask,
                self.attn_dropout_p,
                needs_weights,
                self.training,
            )

        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.attn_dropout_p

        # Check if the mask is causal.
        is_causal_mask: bool = getattr(mask, "is_causal", False)

        attn = F.scaled_dot_product_attention(  # type: ignore[attr-defined]
            queries,
            keys,
            values,
            attn_mask=None if is_causal_mask else mask,
            dropout_p=dropout_p,
            is_causal=is_causal_mask,
        )

        return attn, None


@final
class NaiveSDPA(SDPA):
    """Computes scaled dot-product attention using a non-fused implementation."""

    @finaloverride
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        *,
        mask: Optional[Tensor] = None,
        needs_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return _naive_scaled_dot_product_attention(
            queries,
            keys,
            values,
            mask,
            self.attn_dropout_p,
            needs_weights,
            self.training,
        )


def _naive_scaled_dot_product_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Optional[Tensor],
    dropout_p: float,
    needs_weights: bool,
    training: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    # (*, S, K) @ (*, K, S_kv) = (*, S, S_kv)
    attn_weights = torch.matmul(queries, keys.transpose(-1, -2))

    attn_weights = attn_weights * (queries.size(-1) ** -0.5)

    if mask is not None:
        attn_weights = attn_weights + mask

    # For numerical stability run in single precision.
    attn_weights = softmax(attn_weights, dim=-1, dtype=torch.float32)

    attn_weights = attn_weights.type_as(queries)

    if training and dropout_p > 0.0:
        attn_weights = dropout(attn_weights, dropout_p)

    # (*, S, S_kv) @ (*, S_kv, V) = (*, S, V)
    attn = torch.matmul(attn_weights, values)

    return attn, attn_weights if needs_weights else None


class SDPAFactory(Protocol):
    """Constructs instances of :class:`SDPA`."""

    def __call__(self, *, attn_dropout_p: float) -> SDPA:
        """
        :param attn_dropout_p:
            The dropout probability on attention weights.
        """


def _get_fallback_sdpa_factory() -> SDPAFactory:
    if is_pt2_or_greater():
        return TorchSDPA
    else:
        return NaiveSDPA


_sdpa_factory: SDPAFactory = _get_fallback_sdpa_factory()


def set_default_sdpa(factory: Optional[SDPAFactory]) -> None:
    """Set the process-wide default scaled dot-product attention module.

    If ``None``, defaults to :class:`TorchSDPA` if available; otherwise, to
    :class:`NaiveSDPA`.
    """
    global _sdpa_factory

    if factory is not None:
        _sdpa_factory = factory
    else:
        _sdpa_factory = _get_fallback_sdpa_factory()


@contextmanager
def sdpa(factory: Optional[SDPAFactory]) -> Generator[None, None, None]:
    """Set a temporary default scaled dot-product attention module."""
    original_factory = _sdpa_factory

    set_default_sdpa(factory)

    try:
        yield
    finally:
        set_default_sdpa(original_factory)


def create_default_sdpa(*, attn_dropout_p: float = 0.0) -> SDPA:
    """Create an instance of the default scaled dot-product attention module.

    :param attn_dropout_p:
        The dropout probability on attention weights.
    """
    return _sdpa_factory(attn_dropout_p=attn_dropout_p)
