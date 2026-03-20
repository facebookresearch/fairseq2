# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Protocol

from fairseq2.models.transformer.attention_bias import AttentionBias
from fairseq2.models.transformer.sdpa.base import SDPA
from fairseq2.models.transformer.sdpa.torch import TorchSDPA


class SDPAFactory(Protocol):
    """Constructs instances of :class:`SDPA`."""

    def __call__(
        self, bias: AttentionBias, *, dropout_p: float = 0.0, scale: float | None = None
    ) -> SDPA:
        """
        :param dropout_p: The dropout probability on attention weights.
        :param scale: The scaling factor for attention logits. If ``None``, uses
            default 1/sqrt(head_dim) scaling.
        """


_sdpa_factory: SDPAFactory = TorchSDPA


def set_default_sdpa_factory(factory: SDPAFactory) -> None:
    """Sets the default :class:`SDPA` factory."""
    global _sdpa_factory

    _sdpa_factory = factory


def get_default_sdpa_factory() -> SDPAFactory:
    """Gets the default :class:`SDPA` factory."""
    return _sdpa_factory


def create_default_sdpa(
    bias: AttentionBias, *, dropout_p: float = 0.0, scale: float | None = None
) -> SDPA:
    """Creates an instance of the default :class:`SDPA` class.

    :param bias: The attention bias.
    :param dropout_p: The dropout probability on attention weights.
    :param scale: The scaling factor for attention logits. If ``None``, uses
        default 1/sqrt(head_dim) scaling. Set to 1.0 to disable scaling
        (e.g., when using QK normalization).
    """
    return _sdpa_factory(bias, dropout_p=dropout_p, scale=scale)
