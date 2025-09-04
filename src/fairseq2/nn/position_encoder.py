# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import embedding, interpolate
from torch.nn.parameter import Parameter
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn.batch_layout import BatchLayout
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.utils.module import get_name_or_self
from fairseq2.ops import unsqueeze


class PositionEncoder(Module, ABC):
    """Encodes sequences with positional information."""

    def __init__(self, encoding_dim: int) -> None:
        super().__init__()

        self.encoding_dim = encoding_dim

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """
        Returns a copy of ``seqs`` with positional information encoded.

        :param seqs: The input sequences to encode. *Shape:* :math:`([N],S,*,E)`,
            where :math:`N` is the batch size, :math:`S` is the sequence length,
            :math:`*` is any number of batch dimensions including none, and
            :math:`E` is the dimensionality of the positional encodings.
        :param state_bag: If not ``None``, the encoder will operate in
            incremental decoding mode. The first element in ``seqs`` will be
            considered to be at position :attr:`state_bag.step_nr
            <fairseq2.nn.IncrementalStateBag.step_nr>` instead of 0.

        :raises ValueError: when the sequence length of ``seqs`` exceeds
            :attr:`max_seq_len`.

        :returns: The input sequences with positional information encoded.
            *Shape:* Same as ``seqs``.
        """

    if TYPE_CHECKING:
        __call__ = forward


@final
class SinusoidalPositionEncoder(PositionEncoder):
    """Encodes sequences with fixed sinusoidal positional information."""

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int,
        *,
        _legacy_pad_idx: int | None = None,
        device: Device | None = None,
    ) -> None:
        """
        :param encoding_dim: The dimensionality of positional encodings. The
            last dimension of input sequences is expected to have the same
            dimensionality.
        :param max_seq_len: The maximum allowed length for input sequences.
            Sequences longer than ``max_seq_len`` will cause a :class:`ValueError`.

        :raise ValueError: when ``encoding_dim`` is not even.
        """
        super().__init__(encoding_dim)

        if encoding_dim % 2 != 0:
            raise ValueError(
                f"`encoding_dim` must be even, but is {encoding_dim} instead."
            )

        freqs = torch.empty(
            (max_seq_len + 1, encoding_dim), device=device, dtype=torch.float32
        )

        self.freqs: Tensor

        self.register_buffer("freqs", freqs, persistent=False)

        self.max_seq_len = max_seq_len

        # This is a legacy parameter that should only be set when the encodings
        # must be compatible with fairseq.
        if _legacy_pad_idx is None:
            sin_offset = 0
        else:
            sin_offset = 1 + _legacy_pad_idx

        self.sin_offset = sin_offset

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        self.freqs[0] = 0.0  # pad

        device, dtype = self.freqs.device, self.freqs.dtype

        start_step = self.sin_offset

        # (S)
        steps = torch.arange(
            start_step, start_step + self.max_seq_len, device=device, dtype=dtype
        )

        _fill_sin_freq_table(self.freqs[1:], self.encoding_dim, steps)

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        if not self.training and state_bag is not None:
            start_step = state_bag.step_nr
        else:
            start_step = 0

        max_seq_len = start_step + seqs_layout.max_seq_len

        if max_seq_len > self.max_seq_len:
            raise ValueError(
                f"The lengths of all sequences in `seqs` must be less than or equal to the maximum sequence length ({self.max_seq_len}), but at least one sequence is of length {max_seq_len} instead."
            )

        if seqs_layout.packed or seqs_layout.padded:
            indices = seqs_layout.position_indices + 1  # +1 for padding

            if not self.training and state_bag is not None:
                indices = state_bag.step_nr + indices

            # ([N], S, E)
            freqs = self.freqs[indices]
        else:
            batch_width = seqs_layout.width

            if not self.training and state_bag is not None:
                start_step = 1 + state_bag.step_nr
            else:
                start_step = 1

            # (S, E)
            freqs = self.freqs[start_step : start_step + batch_width]

            # (S, E) -> (1, S, E)
            freqs = freqs.unsqueeze(0)

        if d := seqs.ndim - freqs.ndim:
            freqs = unsqueeze(freqs, dim=-2, count=d)

        fp32_seqs = seqs.float() + freqs

        return fp32_seqs.type_as(seqs)

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"encoding_dim={self.encoding_dim}, max_seq_len={self.max_seq_len}"


def _fill_sin_freq_table(
    freqs: Tensor, encoding_dim: int, steps: Tensor, correction: int = 1
) -> None:
    freqs = freqs.flatten(0, -2)

    num_sin = encoding_dim // 2

    l_half = freqs[:, :num_sin]
    r_half = freqs[:, num_sin:]

    # (E)
    indices = torch.arange(num_sin, device=steps.device, dtype=steps.dtype)

    # This is identical to tensor2tensor's implementation.
    freqs = torch.exp(indices * -math.log(10000.0) / (num_sin - correction))

    # (S) x (E) -> (S, E)
    torch.outer(steps, freqs, out=l_half)

    # The cosine frequencies might be truncated if the table is shorter than the
    # encoding dimension due to rounding.
    r_dim = r_half.size(1)

    r_half.copy_(l_half[:, :r_dim])

    l_half.sin_()
    r_half.cos_()


@final
class LearnedPositionEncoder(PositionEncoder):
    """Encodes sequences with learned positional embeddings."""

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param encoding_dim: The dimensionality of positional encodings. The
            last dimension of input sequences is expected to have the same
            dimensionality.
        :param max_seq_len: The maximum allowed length for input sequences.
            Sequences longer than ``max_seq_len`` will cause a :class:`ValueError`.
        """
        super().__init__(encoding_dim)

        self.weight = Parameter(
            torch.empty((max_seq_len + 1, encoding_dim), device=device, dtype=dtype)
        )

        self.max_seq_len = max_seq_len

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

        with torch.no_grad():
            self.weight[0].fill_(0.0)  # pad

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        if not self.training and state_bag is not None:
            start_step = state_bag.step_nr
        else:
            start_step = 0

        max_seq_len = start_step + seqs_layout.max_seq_len

        if max_seq_len > self.max_seq_len:
            raise ValueError(
                f"The lengths of all sequences in `seqs` must be less than or equal to the maximum sequence length ({self.max_seq_len}), but at least one sequence is of length {max_seq_len} instead."
            )

        indices = seqs_layout.position_indices + 1  # +1 for padding

        if not self.training and state_bag is not None:
            indices = state_bag.step_nr + indices

        # ([N], S, E)
        embeds = embedding(indices, self.weight, padding_idx=0)

        if d := seqs.ndim - embeds.ndim:
            embeds = unsqueeze(embeds, dim=-2, count=d)

        return seqs + embeds

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"encoding_dim={self.encoding_dim}, max_seq_len={self.max_seq_len}"


@final
class RotaryEncoder(PositionEncoder):
    """
    Encodes sequences with relative positional information as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2104.09864`.
    """

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
        :param encoding_dim: The dimensionality of positional encodings. The
            last dimension of input sequences is expected to have the same
            dimensionality.
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
        super().__init__(encoding_dim)

        if encoding_dim % 2 != 0:
            raise ValueError(
                f"`encoding_dim` must be even, but is {encoding_dim} instead."
            )

        freqs = torch.empty(
            (max_seq_len + 1, encoding_dim // 2, 2), device=device, dtype=torch.float32
        )

        self.freqs: Tensor

        self.register_buffer("freqs", freqs, persistent=False)

        self.max_seq_len = max_seq_len

        self.theta = theta

        self.freqs_init_fn = freqs_init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        self.freqs[0] = 0.0  # pad

        device = self.freqs.device

        complex_freqs = torch.view_as_complex(self.freqs[1:])

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
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        if not self.training and state_bag is not None:
            start_step = state_bag.step_nr
        else:
            start_step = 0

        max_seq_len = start_step + seqs_layout.max_seq_len

        if max_seq_len > self.max_seq_len:
            raise ValueError(
                f"The lengths of all sequences in `seqs` must be less than or equal to the maximum sequence length ({self.max_seq_len}), but at least one sequence is of length {max_seq_len} instead."
            )

        complex_freqs = torch.view_as_complex(self.freqs)

        if seqs_layout.packed or seqs_layout.padded:
            indices = seqs_layout.position_indices + 1  # +1 for padding

            if not self.training and state_bag is not None:
                indices = state_bag.step_nr + indices

            # ([N], S, E / 2)
            complex_freqs = complex_freqs[indices]
        else:
            batch_width = seqs_layout.width

            if not self.training and state_bag is not None:
                start_step = 1 + state_bag.step_nr
            else:
                start_step = 1

            # (S, E / 2)
            complex_freqs = complex_freqs[start_step : start_step + batch_width]

            # (S, E / 2) -> (1, S, E / 2)
            complex_freqs = complex_freqs.unsqueeze(0)

        # ([N], S, *, E) -> ([N], S, *, E / 2, 2)
        seqs = seqs.unflatten(-1, (-1, 2))

        # ([N], S, *, E / 2, 2) -> ([N], S, *, E / 2)
        complex_seqs = torch.view_as_complex(seqs.float())

        if d := complex_seqs.ndim - complex_freqs.ndim:
            complex_freqs = unsqueeze(complex_freqs, dim=-2, count=d)

        complex_seqs = complex_seqs * complex_freqs

        # ([N], S, *, E / 2) -> ([N], S, *, E)
        fp32_seqs = torch.view_as_real(complex_seqs).flatten(-2)

        return fp32_seqs.type_as(seqs)

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        s = (
            f"encoding_dim={self.encoding_dim}, "
            f"max_seq_len={self.max_seq_len}, "
            f"theta={self.theta}"
        )

        if self.freqs_init_fn is not None:
            freqs_init_fn = get_name_or_self(self.freqs_init_fn)

            s = f"{s}, freqs_init_fn={freqs_init_fn}"

        return s


@final
class ReferenceRotaryEncoder(PositionEncoder):
    """
    Encodes sequences with relative positional information as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2104.09864`.
    """

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int,
        *,
        theta: float = 10_000.0,
        device: Device | None = None,
    ) -> None:
        """
        :param encoding_dim: The dimensionality of positional encodings. The
            last dimension of input sequences is expected to have the same
            dimensionality.
        :param max_seq_len: The maximum allowed length for input sequences.
            Sequences longer than ``max_seq_len`` will cause a :class:`ValueError`.
        :param theta: The coefficient of the long-term decay as described in
            section 3.3 of the reference paper.

        :raise ValueError: when ``encoding_dim`` is not even.
        """
        super().__init__(encoding_dim)

        if encoding_dim % 2 != 0:
            raise ValueError(
                f"`encoding_dim` must be even, but is {encoding_dim} instead."
            )

        cos_freqs = torch.empty(
            (max_seq_len + 1, encoding_dim), device=device, dtype=torch.float32
        )

        sin_freqs = torch.empty(
            (max_seq_len + 1, encoding_dim), device=device, dtype=torch.float32
        )

        self.cos_freqs: Tensor
        self.sin_freqs: Tensor

        self.register_buffer("cos_freqs", cos_freqs, persistent=False)
        self.register_buffer("sin_freqs", sin_freqs, persistent=False)

        self.max_seq_len = max_seq_len

        self.theta = theta

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        self.cos_freqs[0] = 0.0  # pad
        self.sin_freqs[0] = 0.0  # pad

        dtype = torch.float32

        device = self.cos_freqs.device

        encoding_dim = self.encoding_dim

        # (E)
        indices = torch.arange(encoding_dim // 2, device=device, dtype=dtype)

        # (E) -> (1, E)
        indices = indices.unsqueeze(0)

        # (S)
        steps = torch.arange(self.max_seq_len, device=device, dtype=dtype)

        # (S, 1)
        steps = steps.unsqueeze(1)

        # (S, 1) x (1, E) -> (S, E)
        table = torch.matmul(steps, self.theta ** (-2.0 * indices / encoding_dim))

        cos = torch.cos(table)
        sin = torch.sin(table)

        self.cos_freqs[1:, : encoding_dim // 2] = cos
        self.cos_freqs[1:, encoding_dim // 2 :] = cos

        self.sin_freqs[1:, : encoding_dim // 2] = sin
        self.sin_freqs[1:, encoding_dim // 2 :] = sin

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        if not self.training and state_bag is not None:
            start_step = state_bag.step_nr
        else:
            start_step = 0

        max_seq_len = start_step + seqs_layout.max_seq_len

        if max_seq_len > self.max_seq_len:
            raise ValueError(
                f"The lengths of all sequences in `seqs` must be less than or equal to the maximum sequence length ({self.max_seq_len}), but at least one sequence is of length {max_seq_len} instead."
            )

        if seqs_layout.packed or seqs_layout.padded:
            indices = seqs_layout.position_indices + 1  # +1 for padding

            if not self.training and state_bag is not None:
                indices = state_bag.step_nr + indices

            # ([N], S, E)
            cos_freqs = self.cos_freqs[indices]
            sin_freqs = self.sin_freqs[indices]
        else:
            batch_width = seqs_layout.width

            if not self.training and state_bag is not None:
                start_step = 1 + state_bag.step_nr
            else:
                start_step = 1

            # (S, E)
            cos_freqs = self.cos_freqs[start_step : start_step + batch_width]
            sin_freqs = self.sin_freqs[start_step : start_step + batch_width]

            # (S, E) -> (1, S, E)
            cos_freqs = cos_freqs.unsqueeze(0)
            sin_freqs = sin_freqs.unsqueeze(0)

        if d := seqs.ndim - cos_freqs.ndim:
            cos_freqs = unsqueeze(cos_freqs, dim=-2, count=d)
            sin_freqs = unsqueeze(sin_freqs, dim=-2, count=d)

        fp32_seqs = seqs.float()

        fp32_rotated_seqs = self._rotate_half_way(fp32_seqs)

        fp32_seqs = (fp32_seqs * cos_freqs) + (fp32_rotated_seqs * sin_freqs)

        return fp32_seqs.type_as(seqs)

    def _rotate_half_way(self, seqs: Tensor) -> Tensor:
        half1 = seqs[..., : self.encoding_dim // 2]
        half2 = seqs[..., self.encoding_dim // 2 :]

        return torch.cat((-half2, half1), dim=-1)

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"encoding_dim={self.encoding_dim}, "
            f"max_seq_len={self.max_seq_len}, "
            f"theta={self.theta}"
        )


class InterpolatedPositionEncoder(Module, ABC):
    """Encodes N-dimensional inputs with interpolated positional information."""

    def __init__(self, encoding_dim: int) -> None:
        super().__init__()

        self.encoding_dim = encoding_dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Returns a copy of ``x`` with positional information encoded.

        :params x: The inputs to encode. *Shape:* :math:`(N,*,E)`, where
            :math:`N` is the batch size, :math:`*` is any number of
            implementation-specific dimensions, and :math:`E` is the
            dimensionality of the positional encodings.

        :returns: The inputs with positional information encoded.  *Shape:* Same
            as ``x``.
        """

    if TYPE_CHECKING:
        __call__ = forward


class SinusoidalNdPositionEncoder(InterpolatedPositionEncoder):
    """
    Provides a skeletal implementation of interpolated sinusoidal position
    encoders.
    """

    def __init__(
        self,
        encoding_dim: int,
        grid_dims: tuple[int, ...],
        *,
        device: Device | None = None,
    ) -> None:
        """
        :param encoding_dim: The dimensionality of positional encodings. The
            last dimension of inputs is expected to have the same dimensionality.
        :param grid_dims: The dimensionality of the frequency table.
        """
        super().__init__(encoding_dim)

        if encoding_dim % 2 != 0:
            raise ValueError(
                f"`encoding_dim` must be even, but is {encoding_dim} instead."
            )

        freqs = torch.empty(
            grid_dims + (encoding_dim,), device=device, dtype=torch.float32
        )

        self.grid_dims = grid_dims

        self.freqs: Tensor

        self.register_buffer("freqs", freqs, persistent=False)

    def reset_parameters(self) -> None:
        self.reset_non_persistent_buffers()

    @abstractmethod
    def reset_non_persistent_buffers(self) -> None: ...

    @override
    def forward(self, x: Tensor) -> Tensor:
        freqs = self._interpolate_freqs_as(x)

        fp32_x = x.float() + freqs

        return fp32_x.type_as(x)

    @abstractmethod
    def _interpolate_freqs_as(self, x: Tensor) -> Tensor:
        """
        Interpolates (or extrapolates) the frequency table to the dimensionality
        of ``x``.

        :params x: The inputs to encode. *Shape:* :math:`(N,*,E)`, where
            :math:`N` is the batch size, :math:`*` is the same number of
            dimensions as :attr:`grid_dims`, but potentially with different
            dimensionality, and :math:`E` is the dimensionality of the
            positional encodings.

        :returns: The interpolated (or extrapolated) frequency table. *Shape:*
            Same as ``x``.
        """

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"encoding_dim={self.encoding_dim}, grid_dims={self.grid_dims}"


class Sinusoidal2dPositionEncoder(SinusoidalNdPositionEncoder):
    """
    Encodes 2-dimensional inputs with sinusoidal positional information.

    .. note::
        This implementation uses bicubic interpolation. The interpolation
        technique can be changed by subclassing this type and overriding the
        :meth:`_interpolate_freqs_as` method.
    """

    def __init__(
        self,
        encoding_dim: int,
        grid_dims: tuple[int, int],
        *,
        device: Device | None = None,
    ) -> None:
        """
        :param encoding_dim: The dimensionality of positional encodings. The
            last dimension of inputs is expected to have the same dimensionality.
        :param grid_dims: The dimensionality of the depth, height, and width
            dimensions.
        """
        super().__init__(encoding_dim, grid_dims, device=device)

        self.reset_parameters()

    @override
    def reset_non_persistent_buffers(self) -> None:
        freqs = self.freqs

        device, dtype = freqs.device, freqs.dtype

        h, w = freqs.shape[:-1]

        h_steps = torch.arange(h, device=device, dtype=dtype)
        w_steps = torch.arange(w, device=device, dtype=dtype)

        h_coords, w_coords = torch.meshgrid(h_steps, w_steps, indexing="ij")

        h_coords = h_coords.flatten()
        w_coords = w_coords.flatten()

        uniform_dim = math.ceil(self.encoding_dim / 4) * 2

        h_dim = uniform_dim
        w_dim = uniform_dim

        idx = 0

        _fill_sin_freq_table(
            freqs[..., idx : idx + h_dim], h_dim, h_coords, correction=0
        )

        idx = h_dim

        _fill_sin_freq_table(
            freqs[..., idx : idx + w_dim], w_dim, w_coords, correction=0
        )

    @override
    def _interpolate_freqs_as(self, x: Tensor) -> Tensor:
        freqs = self.freqs

        if x.ndim != 4:
            raise ValueError(
                f"`x` must be 4 dimensional, but is {x.ndim} dimensional instead."
            )

        frq_dims, inp_dims = freqs.shape[:-1], x.shape[1:-1]

        if frq_dims == inp_dims:
            return freqs

        frq_h, frq_w = frq_dims
        inp_h, inp_w = inp_dims

        scale_factor = math.sqrt((inp_h * inp_w) / (frq_h * frq_w))

        # (H_frq, W_frq, E) -> (1, H_frq, W_frq, E)
        freqs = freqs.unsqueeze(0)

        # (1, H_frq, W_frq, E) -> (1, E, H_frq, W_frq)
        freqs = freqs.permute(0, 3, 1, 2)

        # (1, E, H_frq, W_frq) -> (1, E, H_inp, W_inp)
        freqs = interpolate(freqs, scale_factor=scale_factor, mode="bicubic")

        # (1, E, H_inp, W_inp) -> (1, H_inp, W_inp, E)
        freqs = freqs.permute(0, 2, 3, 1)

        # (1, H_inp, W_inp, E) -> (H_inp, W_inp, E)
        return freqs.squeeze(0)  # type: ignore[no-any-return]


class Sinusoidal3dPositionEncoder(SinusoidalNdPositionEncoder):
    """
    Encodes 3-dimensional inputs with sinusoidal positional information.

    .. note::
        This implementation uses trilinear interpolation. The interpolation
        technique can be changed by subclassing this type and overriding the
        :meth:`_interpolate_freqs_as` method.
    """

    def __init__(
        self,
        encoding_dim: int,
        grid_dims: tuple[int, int, int],
        *,
        uniform_power: bool = False,
        device: Device | None = None,
    ) -> None:
        """
        :param encoding_dim: The dimensionality of positional encodings. The
            last dimension of inputs is expected to have the same dimensionality.
        :param grid_dims: The dimensionality of the depth, height, and width
            dimensions.
        :param uniform_power: If ``True``, each dimension of ``grid_dims`` will
            have equal representation in the produced positional encodings. This
            means, if ``True``, a positional encoding will consists of 1/3 depth,
            1/3 height, and 1/3 width information; otherwise, 1/2 depth, 1/4
            height, and 1/4 width information.
        """
        super().__init__(encoding_dim, grid_dims, device=device)

        self.uniform_power = uniform_power

        self.reset_parameters()

    @override
    def reset_non_persistent_buffers(self) -> None:
        freqs = self.freqs

        device, dtype = freqs.device, freqs.dtype

        d, h, w = freqs.shape[:-1]

        d_steps = torch.arange(d, device=device, dtype=dtype)
        h_steps = torch.arange(h, device=device, dtype=dtype)
        w_steps = torch.arange(w, device=device, dtype=dtype)

        d_coords, h_coords, w_coords = torch.meshgrid(
            d_steps, h_steps, w_steps, indexing="ij"
        )

        d_coords = d_coords.flatten()
        h_coords = h_coords.flatten()
        w_coords = w_coords.flatten()

        if self.uniform_power:
            uniform_dim = math.ceil(self.encoding_dim / 6) * 2

            d_dim = uniform_dim
            h_dim = uniform_dim
            w_dim = uniform_dim
        else:
            d_dim = math.ceil(self.encoding_dim / 4) * 2
            h_dim = math.ceil(self.encoding_dim / 8) * 2
            w_dim = math.ceil(self.encoding_dim / 8) * 2

        idx = 0

        _fill_sin_freq_table(
            freqs[..., idx : idx + d_dim], d_dim, d_coords, correction=0
        )

        idx = d_dim

        _fill_sin_freq_table(
            freqs[..., idx : idx + h_dim], h_dim, h_coords, correction=0
        )

        idx = d_dim + h_dim

        _fill_sin_freq_table(
            freqs[..., idx : idx + w_dim], w_dim, w_coords, correction=0
        )

    @override
    def _interpolate_freqs_as(self, x: Tensor) -> Tensor:
        freqs = self.freqs

        if x.ndim != 5:
            raise ValueError(
                f"`x` must be 5 dimensional, but is {x.ndim} dimensional instead."
            )

        frq_dims, inp_dims = freqs.shape[:-1], x.shape[1:-1]

        if frq_dims == inp_dims:
            return freqs

        frq_d, frq_h, frq_w = frq_dims
        inp_d, inp_h, inp_w = inp_dims

        scale_factor = (inp_d / frq_d, inp_h / frq_h, inp_w / frq_w)

        # (D_frq, H_frq, W_frq, E) -> (1, D_frq, H_frq, W_frq, E)
        freqs = freqs.unsqueeze(0)

        # (1, D_frq, H_frq, W_frq, E) -> (1, E, D_frq, H_frq, W_frq)
        freqs = freqs.permute(0, 4, 1, 2, 3)

        # (1, E, D_frq, H_frq, W_frq) -> (1, E, D_inp, H_inp, W_inp)
        freqs = interpolate(freqs, scale_factor=scale_factor, mode="trilinear")

        # (1, E, D_inp, H_inp, W_inp) -> (1, D_inp, H_inp, W_inp, E)
        freqs = freqs.permute(0, 2, 3, 4, 1)

        # (1, D_inp, H_inp, W_inp, E) -> (D_inp, H_inp, W_inp, E)
        return freqs.squeeze(0)  # type: ignore[no-any-return]
