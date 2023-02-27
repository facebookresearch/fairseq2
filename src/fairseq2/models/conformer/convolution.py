# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Conv1d, Module

from fairseq2.nn.utils.fn import get_name


class ConformerConvolution(Module):
    """Represents a Conformer convolution module as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2005.08100`."""

    model_dim: int
    pointwise_conv1d_1: Conv1d
    depthwise_conv1d: Conv1d
    batch_norm: BatchNorm1d
    depthwise_activation_fn: Callable[[Tensor], Tensor]
    pointwise_conv1d_2: Conv1d

    def __init__(
        self,
        model_dim: int,
        depthwise_kernel_size: int,
        depthwise_activation_fn: Optional[Callable[[Tensor], Tensor]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs).
        :param depthwise_kernel_size:
            The kernel size of the depthwise convolution.
        :param depthwise_activation_fn:
            The activation to apply to outputs of the depthwise convolution. If
            ``None``, :func:`~torch.nn.functional.silu` (a.k.a. swish) will be
            used.
        """
        super().__init__()

        self.model_dim = model_dim

        # We treat the model size as the number of input channels to the
        # first pointwise convolution.
        self.pointwise_conv1d_1 = Conv1d(
            in_channels=model_dim,
            # We apply GLU to outputs to bring them back to model_dim.
            out_channels=2 * model_dim,
            kernel_size=1,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.depthwise_conv1d = Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=depthwise_kernel_size,
            # We preserve the sequence length regardless of the kernel size.
            padding="same",
            # We want to perform depthwise convolution.
            groups=model_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.batch_norm = BatchNorm1d(model_dim, device=device, dtype=dtype)

        if depthwise_activation_fn is None:
            self.depthwise_activation_fn = F.silu  # a.k.a. swish
        else:
            self.depthwise_activation_fn = depthwise_activation_fn

        self.pointwise_conv1d_2 = Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
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
            sequence length, and :math:`M` is the model size.

        :returns:
            The processed output. *Shape:* Same as ``x``.
        """
        # (N, S, M) -> (N, M, S)
        x = x.transpose(1, 2)

        # This is mathematically equivalent to a dot-product.
        # (N, M, S) -> (N, 2 * M, S)
        x = self.pointwise_conv1d_1(x)

        # (N, 2 * M, S) -> (N, M, S)
        x = F.glu(x, dim=1)

        # (N, M, S) -> (N, M, S)
        x = self.depthwise_conv1d(x)

        x = self.batch_norm(x)

        x = self.depthwise_activation_fn(x)

        # This is mathematically equivalent to a dot-product.
        # (N, M, S) -> (N, M, S)
        x = self.pointwise_conv1d_2(x)

        # (N, M, S) -> (N, S, M)
        return x.transpose(1, 2)

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}, depthwise_activation_fn={get_name(self.depthwise_activation_fn)}"
