# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import embedding
from torch.nn.parameter import Parameter
from typing_extensions import override

from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import DataType, Device


class PositionEncoder(Module, ABC):
    """Encodes sequences with positional information."""

    encoding_dim: int
    max_seq_len: int | None

    def __init__(self, encoding_dim: int, max_seq_len: int | None) -> None:
        """
        :param encoding_dim: The dimensionality of positional encodings. Input
            sequences are expected to have the same dimensionality.
        :param max_seq_len: The maximum allowed length for input sequences.
            Sequences longer than ``max_seq_len`` will cause a :class:`ValueError`.
            Typically it is set to the context length of the underlying model.
            If ``None``, sequences can have arbitrary length.
        """
        super().__init__()

        self.encoding_dim = encoding_dim
        self.max_seq_len = max_seq_len

    def forward(
        self,
        seqs: Tensor,
        padding_mask: PaddingMask | None,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """
        Returns a copy of ``seqs`` with positional information encoded.

        :param seqs: The input sequences to encode. *Shape:* :math:`(*,S,E)`,
            where :math:`*` is any number of batch dimensions including none,
            :math:`S` is the sequence length, and :math:`E` is the dimensionality
            of the positional encodings.
        :param padding_mask: The padding mask of ``seqs``. *Shape:* :math:`(*,S)`,
            where :math:`*` is any number of batch dimensions including none and
            :math:`S` is the sequence length.
        :param state_bag: If not ``None``, the encoder will operate in
            incremental decoding mode. This means that the first step in ``seqs``
            will be considered to be at position :attr:`state_bag.step_nr
            <fairseq2.nn.IncrementalStateBag.step_nr>` instead of 0.

        :raises ValueError: when the sequence length of ``seqs`` exceeds
            :attr:`max_seq_len`.

        :returns: The input sequences with positional information encoded.
            *Shape:* Same as ``seqs``.
        """
        if self.max_seq_len is not None:
            if self.training or state_bag is None:
                start_step = 0
            else:
                start_step = state_bag.step_nr

            if (seq_len := start_step + seqs.size(-2)) > self.max_seq_len:
                raise ValueError(
                    f"The input sequence length must be less than or equal to the maximum sequence length ({self.max_seq_len}), but is {seq_len} instead."
                )

        return self._do_forward(seqs, padding_mask, state_bag)

    @abstractmethod
    def _do_forward(
        self,
        seqs: Tensor,
        padding_mask: PaddingMask | None,
        state_bag: IncrementalStateBag | None,
    ) -> Tensor:
        """
        When overriden in a subclass, returns a copy of ``seqs`` with positional
        information encoded. See :meth:`forward` for parameter descriptions.

        :meta public:
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        s = f"encoding_dim={self.encoding_dim}"

        if self.max_seq_len is not None:
            s = f"{s}, max_seq_len={self.max_seq_len}"

        return s


@final
class SinusoidalPositionEncoder(PositionEncoder):
    """Encodes sequences with fixed sinusoidal positional information."""

    freqs: Tensor

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int,
        *,
        _legacy_pad_idx: int | None = None,
        device: Device | None = None,
    ) -> None:
        """
        :param encoding_dim: The dimensionality of positional encodings. Input
            sequences are expected to have the same dimensionality.
        :param max_seq_len: The maximum allowed length for input sequences.
            Sequences longer than ``max_seq_len`` will cause a :class:`ValueError`.

        :raise ValueError: when ``encoding_dim`` is not even.
        """
        super().__init__(encoding_dim, max_seq_len)

        if encoding_dim % 2 != 0:
            raise ValueError(
                f"`encoding_dim` must be even, but is {encoding_dim} instead."
            )

        # This is a legacy parameter that should only be set when the encodings
        # must be compatible with fairseq.
        if _legacy_pad_idx is None:
            self._sin_offset = 0
        else:
            self._sin_offset = 1 + _legacy_pad_idx

        freqs = torch.empty(
            (max_seq_len, encoding_dim), device=device, dtype=torch.float32
        )

        self.register_buffer("freqs", freqs, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        """Reset the non-persistent buffers of the module."""
        assert self.max_seq_len is not None

        device, dtype = self.freqs.device, self.freqs.dtype

        num_sin = self.encoding_dim // 2

        l_half = self.freqs[:, :num_sin]
        r_half = self.freqs[:, num_sin:]

        start_step = self._sin_offset

        # (S)
        steps = torch.arange(
            start_step, start_step + self.max_seq_len, device=device, dtype=dtype
        )

        # (E)
        indices = torch.arange(num_sin, device=device, dtype=dtype)

        # This is identical to tensor2tensor's implementation.
        freqs = torch.exp(indices * -math.log(10000.0) / (num_sin - 1))

        # (S) x (E) -> (S, E)
        torch.outer(steps, freqs, out=l_half)

        r_half.copy_(l_half)

        l_half.sin_()
        r_half.cos_()

    @override
    def _do_forward(
        self,
        seqs: Tensor,
        padding_mask: PaddingMask | None,
        state_bag: IncrementalStateBag | None,
    ) -> Tensor:
        """:meta private:"""
        seq_len = seqs.size(-2)

        if self.training or state_bag is None:
            start_step = 0
        else:
            start_step = state_bag.step_nr

        fp32_seqs = seqs.float() + self.freqs[start_step : start_step + seq_len]

        return fp32_seqs.type_as(seqs)


@final
class LearnedPositionEncoder(PositionEncoder):
    """Encodes sequences with learned positional embeddings."""

    weight: Parameter

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param encoding_dim: The dimensionality of positional encodings. Input
            sequences are expected to have the same dimensionality.
        :param max_seq_len: The maximum allowed length for input sequences.
            Sequences longer than ``max_seq_len`` will cause a :class:`ValueError`.
        """
        super().__init__(encoding_dim, max_seq_len)

        self.weight = Parameter(
            torch.empty((max_seq_len, encoding_dim), device=device, dtype=dtype)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.normal_(self.weight)

    @override
    def _do_forward(
        self,
        seqs: Tensor,
        padding_mask: PaddingMask | None,
        state_bag: IncrementalStateBag | None,
    ) -> Tensor:
        """:meta private:"""
        seq_len = seqs.size(-2)

        if self.training or state_bag is None:
            start_step = 0
        else:
            start_step = state_bag.step_nr

        steps = torch.arange(
            start_step, start_step + seq_len, device=seqs.device, dtype=torch.int64
        )

        return seqs + embedding(steps, self.weight)


@final
class RotaryEncoder(PositionEncoder):
    """
    Encodes sequences with relative positional information as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2104.09864`.
    """

    freqs: Tensor
    theta: float
    freqs_init_fn: Callable[[RotaryEncoder], Tensor] | None

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int,
        *,
        theta: float = 10_000.0,
        freqs_init_fn: Callable[[RotaryEncoder], Tensor] | None = None,
        device: Device | None = None,
    ) -> None:
        """
        :param encoding_dim: The dimensionality of positional encodings. Input
            sequences are expected to have the same dimensionality.
        :param max_seq_len: The maximum allowed length for input sequences.
            Sequences longer than ``max_seq_len`` will cause a :class:`ValueError`.
        :param theta: The coefficient of the long-term decay as described in
            section 3.3 of the reference paper.
        :param freqs_init_fn: A callable to initialize the frequency table. The
            encoder will be passed to the callable as an argument and it is
            expected for the callable to return a :class:`~torch.Tensor` holding
            the frequency table. If ``None``, the frequencies will be initialized
            as described in the reference paper.

        :raise ValueError: when ``encoding_dim`` is not even.
        """
        super().__init__(encoding_dim, max_seq_len)

        if encoding_dim % 2 != 0:
            raise ValueError(
                f"`encoding_dim` must be even, but is {encoding_dim} instead."
            )

        freqs = torch.empty(
            (max_seq_len, encoding_dim // 2, 2), device=device, dtype=torch.float32
        )

        self.register_buffer("freqs", freqs, persistent=False)

        self.theta = theta
        self.freqs_init_fn = freqs_init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        """Reset the non-persistent buffers of the module."""
        assert self.max_seq_len is not None

        device = self.freqs.device

        complex_freqs = torch.view_as_complex(self.freqs)

        # (S)
        steps = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)

        if self.freqs_init_fn is None:
            # (E / 2)
            indices = torch.arange(
                0, self.encoding_dim, step=2, device=device, dtype=torch.float32
            )

            freqs = 1.0 / (self.theta ** (indices / self.encoding_dim))
        else:
            freqs = self.freqs_init_fn(self)

        # (S) x (E / 2) -> (S, E / 2)
        freqs = torch.outer(steps, freqs)

        # (S, E / 2)
        torch.polar(torch.ones_like(freqs), freqs, out=complex_freqs)

    @override
    def _do_forward(
        self,
        seqs: Tensor,
        padding_mask: PaddingMask | None,
        state_bag: IncrementalStateBag | None,
    ) -> Tensor:
        """:meta private:"""
        seq_len = seqs.size(-2)

        if self.training or state_bag is None:
            start_step = 0
        else:
            start_step = state_bag.step_nr

        complex_freqs = torch.view_as_complex(self.freqs)

        complex_freqs = complex_freqs[start_step : start_step + seq_len]

        # (*, S, E) -> (*, S, E / 2, 2)
        seqs = seqs.unflatten(-1, (-1, 2))

        complex_seqs = torch.view_as_complex(seqs.float())

        complex_seqs = complex_seqs * complex_freqs

        # (*, S, E / 2, 2) -> (*, S, E)
        fp32_seqs = torch.view_as_real(complex_seqs).flatten(-2)

        return fp32_seqs.type_as(seqs)
