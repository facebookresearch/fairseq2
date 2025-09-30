# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, final

from torch import Tensor
from torch.nn import Conv2d, Conv3d, Module
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device


class PatchFeatureExtractor(Module, ABC):
    """
    Extracts patch features from N-dimensional inputs and embeds them in a
    latent space.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: The inputs from which to extract patch features. *Shape:*
            :math:`(N,C,*)`, where :math:`N` is the batch size, :math:`C` is the
            number of channels, and :math:`*` is any number of input-specific
            dimensions.

        :returns: The extracted patch features. *Shape:* :math:`(N,*,E)`, where
              :math:`N` is the batch size, :math:`*` is the same number of
              dimensions as in input, but potentially with different
              dimensionality, and :math:`E` is the dimensionality of the patch
              features.
        """

    if TYPE_CHECKING:
        __call__ = forward


@final
class Conv2dPatchFeatureExtractor(PatchFeatureExtractor):
    """Extracts patch features from 2-dimensional inputs using convolution."""

    def __init__(
        self,
        num_channels: int,
        feature_dim: int,
        patch_dims: tuple[int, int],
        *,
        init_fn: Callable[[Conv2d], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param num_channels: The number of input channels.
        :param feature_dim: The dimensionality of extracted patch features.
        :param patch_dims: The dimensionality of height and width patch
            dimensions.
        """
        super().__init__()

        self.conv = Conv2d(
            num_channels,
            feature_dim,
            kernel_size=patch_dims,
            stride=patch_dims,
            device=device,
            dtype=dtype,
        )

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_fn is not None:
            self.init_fn(self.conv)
        else:
            self.conv.reset_parameters()

    @override
    def forward(self, x: Tensor) -> Tensor:
        # (N, C, H_inp, W_inp) -> (N, H_out, W_out, E)
        return self.conv(x).permute(0, 2, 3, 1)  # type: ignore[no-any-return]


@final
class Conv3dPatchFeatureExtractor(PatchFeatureExtractor):
    """Extracts patch features from 3-dimensional inputs using convolution."""

    def __init__(
        self,
        num_channels: int,
        feature_dim: int,
        patch_dims: tuple[int, int, int],
        *,
        init_fn: Callable[[Conv3d], None] | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param num_channels: The number of input channels.
        :param feature_dim: The dimensionality of extracted patch features.
        :param patch_dims: The dimensionality of depth, height, and width patch
            dimensions.
        """
        super().__init__()

        self.conv = Conv3d(
            num_channels,
            feature_dim,
            kernel_size=patch_dims,
            stride=patch_dims,
            device=device,
            dtype=dtype,
        )

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_fn is not None:
            self.init_fn(self.conv)
        else:
            self.conv.reset_parameters()

    @override
    def forward(self, x: Tensor) -> Tensor:
        # (N, C, D_inp, H_inp, W_inp) -> (N, D_out, H_out, W_out, E)
        return self.conv(x).permute(0, 2, 3, 4, 1)  # type: ignore[no-any-return]
