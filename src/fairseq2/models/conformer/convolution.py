# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from torch import Tensor
from torch.nn import GLU, BatchNorm1d, Conv1d, Module, SiLU

from fairseq2.nn.utils.mask import apply_padding_mask
from fairseq2.typing import DataType, Device


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
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
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

        self.pointwise_conv1_activation = GLU(dim=1)

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

    def forward(self, seqs: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
        """
        :param seqs:
            The sequences to process. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.
        :param padding_mask:
            The float padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math:`S` is the sequence length.

        :returns:
            The processed sequences. *Shape:* Same as ``seqs``.
        """
        # Ensure that we do not leak padded positions in depthwise convolution.
        seqs = apply_padding_mask(seqs, padding_mask)

        # (N, S, M) -> (N, M, S)
        seqs = seqs.transpose(1, 2)

        # This is mathematically equivalent to a dot-product.
        # (N, M, S) -> (N, 2 * M, S)
        seqs = self.pointwise_conv1(seqs)

        # (N, 2 * M, S) -> (N, M, S)
        seqs = self.pointwise_conv1_activation(seqs)

        # (N, M, S) -> (N, M, S)
        seqs = self.depthwise_conv(seqs)

        seqs = self.batch_norm(seqs)

        seqs = self.depthwise_activation(seqs)

        # This is mathematically equivalent to a dot-product.
        # (N, M, S) -> (N, M, S)
        seqs = self.pointwise_conv2(seqs)

        # (N, M, S) -> (N, S, M)
        seqs = seqs.transpose(1, 2)

        return seqs

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"
