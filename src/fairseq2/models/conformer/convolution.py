# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import Tensor
from torch.nn import GLU, BatchNorm1d, Conv1d, Module, SiLU


class ConformerConvolution(Module):
    """Represents a Conformer convolution module as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2005.08100`."""

    model_dim: int
    pointwise_conv1: Conv1d
    pointwise_conv1_activation: GLU
    depthwise_conv: Conv1d
    batch_norm: BatchNorm1d
    depthwise_activation: Module
    pointwise_conv2: Conv1d

    def __init__(
        self,
        model_dim: int,
        depthwise_kernel_size: int,
        depthwise_activation: Optional[Module] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs).
        :param depthwise_kernel_size:
            The kernel size of the depthwise convolution.
        :param depthwise_activation:
            The activation to apply to outputs of the depthwise convolution. If
            ``None``, :func:`~torch.nn.SiLU` (a.k.a. swish) will be used.
        """
        super().__init__()

        self.model_dim = model_dim

        # We treat the dimensionality of the model as the number of input
        # channels to the first pointwise convolution.
        self.pointwise_conv1 = Conv1d(
            model_dim,
            # We apply GLU to outputs to bring them back to `model_dim`.
            model_dim * 2,
            kernel_size=1,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.pointwise_conv1_activation = GLU(dim=-2)

        self.depthwise_conv = Conv1d(
            model_dim,
            model_dim,
            depthwise_kernel_size,
            # We preserve the sequence length regardless of the kernel size.
            padding="same",
            # We want to perform depthwise convolution.
            groups=model_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.batch_norm = BatchNorm1d(model_dim, device=device, dtype=dtype)

        if depthwise_activation is None:
            self.depthwise_activation = SiLU()  # a.k.a. swish
        else:
            self.depthwise_activation = depthwise_activation

        self.pointwise_conv2 = Conv1d(
            model_dim,
            model_dim,
            kernel_size=1,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input to process. *Shape:* :math:`(N,S,M)`, or :math:`(S,M)`
            when unbatched, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`M` is the dimensionality of the model.

        :returns:
            The processed output of ``x``. *Shape:* Same as ``x``.
        """
        # (N, S, M) -> (N, M, S)
        x = x.transpose(-1, -2)

        # This is mathematically equivalent to a dot-product.
        # (N, M, S) -> (N, 2 * M, S)
        x = self.pointwise_conv1(x)

        # (N, 2 * M, S) -> (N, M, S)
        x = self.pointwise_conv1_activation(x)

        # (N, M, S) -> (N, M, S)
        x = self.depthwise_conv(x)

        if x.dim() == 3:
            x = self.batch_norm(x)

        x = self.depthwise_activation(x)

        # This is mathematically equivalent to a dot-product.
        # (N, M, S) -> (N, M, S)
        x = self.pointwise_conv2(x)

        # (N, M, S) -> (N, S, M)
        x = x.transpose(-1, -2)

        return x

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"
