# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast, final

import torch.nn as nn
from torch import Tensor
from torch.nn import GELU, Conv1d, Module, Sequential
from torch.nn.utils import remove_weight_norm, weight_norm  # type: ignore[attr-defined]
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.error import NotSupportedError
from fairseq2.nn import (
    BatchLayout,
    IncrementalStateBag,
    PositionEncoder,
    StandardLayerNorm,
)
from fairseq2.nn.utils.mask import apply_mask


@final
class Wav2Vec2PositionEncoder(PositionEncoder):
    """Encodes sequences with relative positional information as described in
    Section 2 of :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    def __init__(
        self,
        model_dim: int,
        kernel_size: int,
        num_groups: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param kernel_size:
            The kernel size of the 1D convolution.
        :param num_groups:
            The number of convolution groups.
        """
        super().__init__(model_dim)

        self.conv = Wav2Vec2PositionalConv1d(
            model_dim,
            model_dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=num_groups,
            device=device,
            dtype=dtype,
        )

        self.remove_pad = kernel_size % 2 == 0

        self.activation = GELU()

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """:meta private:"""
        if state_bag is not None:
            raise NotSupportedError(
                f"`{Wav2Vec2PositionEncoder}` does not support incremental decoding."
            )

        if seqs_layout.packed:
            raise ValueError("`seqs` must not be a packed batch.")

        if seqs_layout.padded:
            padding_mask = seqs_layout.position_indices >= 0

            # We have to ensure that the padded elements are correctly set to
            # zero; otherwise, noise will leak into the feature maps.
            seqs = apply_mask(seqs, padding_mask)

        # (N, S, E) -> (N, E, S)
        encodings = seqs.transpose(1, 2)

        # (N, E, S) -> (N, E, S)
        encodings = self.conv(encodings)

        if self.remove_pad:
            encodings = encodings[:, :, :-1]

        encodings = self.activation(encodings)

        # (N, E, S) -> (N, S, E)
        encodings = encodings.transpose(1, 2)

        return seqs + encodings  # type: ignore[no-any-return]

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"encoding_dim={self.encoding_dim}"


@final
class Wav2Vec2PositionalConv1d(Conv1d):
    """Represents the convolution used in :class:`Wav2Vec2PositionEncoder`."""

    @override
    def reset_parameters(self) -> None:
        model_dim, kernel_size = self.in_channels, self.kernel_size[0]

        try:
            weight = self.weight_g

            remove_weight_norm(self)
        # Raised during the `__init__` call since we don't have the weight norm
        # hook registered yet. Safe to ignore.
        except AttributeError:
            weight = self.weight

        nn.init.normal_(
            self.weight, mean=0.0, std=(4.0 / (kernel_size * model_dim)) ** 0.5
        )

        if not getattr(self, "no_parametrization", False):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message=r".*deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.*"  # fmt: skip
                )

                weight_norm(self, dim=2)

            requires_grad = cast(bool, weight.requires_grad)

            self.weight_v.requires_grad_(requires_grad)
            self.weight_g.requires_grad_(requires_grad)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


@final
class Wav2Vec2StackedPositionEncoder(PositionEncoder):
    """Encodes sequences with relative positional information using a stack
    of 1D convolutions.

    This position encoder is not mentioned in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`, but exists in the
    reference implementation.
    """

    def __init__(
        self,
        model_dim: int,
        kernel_size: int,
        num_groups: int,
        num_layers: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param kernel_size:
            The total kernel size of the 1D convolutions. Each convolution uses
            a kernel size of ``max(3, kernel_size // num_layers)``.
        :param num_groups:
            The number of convolution groups.
        :param num_layers:
            The number of convolution layers.
        """
        super().__init__(model_dim)

        k = max(3, kernel_size // num_layers)

        self.layers = Sequential()

        for _ in range(num_layers):
            layer = Wav2Vec2PositionEncoderLayer(
                model_dim,
                k,
                num_groups,
                device=device,
                dtype=dtype,
            )

            self.layers.append(layer)

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """:meta private:"""
        if state_bag is not None:
            raise NotSupportedError(
                f"`{Wav2Vec2StackedPositionEncoder}` does not support incremental decoding."
            )

        if seqs_layout.packed:
            raise ValueError("`seqs` must not be a packed batch.")

        if seqs_layout.padded:
            padding_mask = seqs_layout.position_indices >= 0

            # We have to ensure that the padded elements are correctly set to
            # zero; otherwise, noise will leak into the feature maps.
            seqs = apply_mask(seqs, padding_mask)

        # (N, S, E) -> (N, E, S)
        encodings = seqs.transpose(1, 2)

        # (N, E, S) -> (N, E, S)
        encodings = self.layers(encodings)

        # (N, E, S) -> (N, S, E)
        encodings = encodings.transpose(1, 2)

        return seqs + encodings  # type: ignore[no-any-return]

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"encoding_dim={self.encoding_dim}"


@final
class Wav2Vec2PositionEncoderLayer(Module):
    """Represents a layer used in :class:`Wav2Vec2StackedPositionEncoder`."""

    def __init__(
        self,
        model_dim: int,
        kernel_size: int,
        num_groups: int,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.conv = Conv1d(
            model_dim,
            model_dim,
            kernel_size,
            padding="same",
            groups=num_groups,
            device=device,
            dtype=dtype,
        )

        self.layer_norm = StandardLayerNorm(
            model_dim, bias=True, elementwise_affine=False, device=device, dtype=dtype
        )

        self.activation = GELU()

    def forward(self, encodings: Tensor) -> Tensor:
        # (N, E, S) -> (N, E, S)
        encodings = self.conv(encodings)

        # (N, E, S) -> (N, S, E)
        encodings = encodings.transpose(1, 2)

        encodings = self.layer_norm(encodings)

        # (N, S, E) -> (N, E, S)
        encodings = encodings.transpose(1, 2)

        encodings = self.activation(encodings)

        return encodings

    if TYPE_CHECKING:
        __call__ = forward
