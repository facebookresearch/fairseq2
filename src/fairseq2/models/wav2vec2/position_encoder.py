# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, final

import torch.nn as nn
from torch import Tensor
from torch.nn import GELU, Conv1d, Module, Sequential
from torch.nn.utils.weight_norm import remove_weight_norm, weight_norm

from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.normalization import LayerNorm, StandardLayerNorm
from fairseq2.nn.padding import PaddingMask, apply_padding_mask
from fairseq2.nn.position_encoder import PositionEncoder
from fairseq2.typing import DataType, Device, finaloverride, override


@final
class Wav2Vec2PositionEncoder(PositionEncoder):
    """Encodes sequences with relative positional information as described in
    Section 2 of :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    conv: Conv1d
    remove_pad: bool
    activation: GELU

    def __init__(
        self,
        model_dim: int,
        kernel_size: int,
        num_groups: int,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param kernel_size:
            The kernel size of the 1D convolution.
        :param num_groups:
            The number of convolution groups.
        """
        super().__init__(model_dim, max_seq_len=None)

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

    @finaloverride
    def _do_forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        """:meta private:"""
        if state_bag is not None:
            raise ValueError(
                "`Wav2Vec2PositionEncoder` does not support incremental decoding."
            )

        # We have to ensure that the padded elements are correctly set to
        # zero; otherwise, noise will leak into the feature maps.
        seqs = apply_padding_mask(seqs, padding_mask)

        # (N, S, E) -> (N, E, S)
        encodings = seqs.transpose(1, 2)

        # (N, E, S) -> (N, E, S)
        encodings = self.conv(encodings)

        if self.remove_pad:
            encodings = encodings[:, :, :-1]

        encodings = self.activation(encodings)

        # (N, E, S) -> (N, S, E)
        encodings = encodings.transpose(1, 2)

        return seqs + encodings


class Wav2Vec2PositionalConv1d(Conv1d):
    """Represents the convolution used in :class:`Wav2Vec2PositionEncoder`."""

    @override
    def reset_parameters(self) -> None:
        model_dim, kernel_size = self.in_channels, self.kernel_size[0]

        try:
            remove_weight_norm(self)
        except ValueError:
            # Raised during the `__init__` call since we don't have the weight
            # norm hook registered yet. Safe to ignore.
            pass

        nn.init.normal_(
            self.weight, mean=0.0, std=(4.0 / (kernel_size * model_dim)) ** 0.5
        )

        weight_norm(self, dim=2)

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

    layers: Sequential

    def __init__(
        self,
        model_dim: int,
        kernel_size: int,
        num_groups: int,
        num_layers: int,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
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
        super().__init__(model_dim, max_seq_len=None)

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

    @finaloverride
    def _do_forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        """:meta private:"""
        if state_bag is not None:
            raise ValueError(
                "`Wav2Vec2StackedPositionEncoder` does not support incremental decoding."
            )

        # We have to ensure that the padded elements are correctly set to
        # zero; otherwise, noise will leak into the feature maps.
        seqs = apply_padding_mask(seqs, padding_mask)

        # (N, S, E) -> (N, E, S)
        encodings = seqs.transpose(1, 2)

        # (N, E, S) -> (N, E, S)
        encodings = self.layers(encodings)

        # (N, E, S) -> (N, S, E)
        encodings = encodings.transpose(1, 2)

        return seqs + encodings


class Wav2Vec2PositionEncoderLayer(Module):
    """Represents a layer used in :class:`Wav2Vec2StackedPositionEncoder`."""

    conv: Conv1d
    layer_norm: LayerNorm
    activation: GELU

    def __init__(
        self,
        model_dim: int,
        kernel_size: int,
        num_groups: int,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
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
