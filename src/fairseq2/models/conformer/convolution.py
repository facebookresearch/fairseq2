# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from torch import Tensor
from torch.nn import GLU, BatchNorm1d, Conv1d, Module, SiLU
from torch.nn.functional import pad
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.error import InternalError
from fairseq2.nn import BatchLayout, StandardLayerNorm
from fairseq2.nn.utils.mask import apply_mask


@final
class ConformerConvolution(Module):
    """Represents a Conformer convolution module as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2005.08100`."""

    def __init__(
        self,
        model_dim: int,
        depthwise_kernel_size: int,
        *,
        causal_depthwise_conv: bool = False,
        norm_type: Literal["batch_norm", "layer_norm"] = "batch_norm",
        depthwise_activation: Module | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param depthwise_kernel_size:
            The kernel size of the depthwise convolution.
        :param causal_depthwise_conv:
            If ``True``, uses a causal depthwise convolution similar to that
            described in Section 2.1 of :cite:t:`https://doi.org/10.48550/arxiv.1609.03499`.
        :param norm_type:
            The type of normalization to apply after the depthwise convolution.
        :param depthwise_activation:
            The activation to apply to outputs of the depthwise convolution. If
            ``None``, :func:`~torch.nn.SiLU` (a.k.a. swish) will be used.
        """
        super().__init__()

        # We treat the dimensionality of the model as the number of input
        # channels to the first pointwise convolution.
        self.pointwise_conv1 = Conv1d(
            model_dim,
            model_dim * 2,  # with GLU brings outputs back to `model_dim`.
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
            padding="same" if not causal_depthwise_conv else 0,
            groups=model_dim,  # depthwise
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.causal_depthwise_conv = causal_depthwise_conv

        if norm_type not in ("batch_norm", "layer_norm"):
            raise ValueError(
                f"`norm_type` must be 'batch_norm' or 'layer_norm', but is '{norm_type}' instead."
            )

        if norm_type == "batch_norm":
            batch_norm = BatchNorm1d(model_dim, device=device, dtype=dtype)
        else:
            batch_norm = None

        self.batch_norm: BatchNorm1d | None

        self.register_module("batch_norm", batch_norm)

        if norm_type == "layer_norm":
            layer_norm = StandardLayerNorm(
                model_dim, bias=True, device=device, dtype=dtype
            )
        else:
            layer_norm = None

        self.layer_norm: StandardLayerNorm | None

        self.register_module("layer_norm", layer_norm)

        if depthwise_activation is not None:
            self.depthwise_activation = depthwise_activation
        else:
            self.depthwise_activation = SiLU()  # a.k.a. swish

        self.pointwise_conv2 = Conv1d(
            model_dim, model_dim, kernel_size=1, bias=False, device=device, dtype=dtype
        )

    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> Tensor:
        """
        :param seqs:
            The sequences to process. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.

        :returns:
            The processed sequences. *Shape:* Same as ``seqs``.
        """
        if seqs_layout.packed:
            raise ValueError("`seqs` must not be a packed batch.")

        if seqs_layout.padded:
            padding_mask = seqs_layout.position_indices >= 0

            # We have to ensure that the padded elements are correctly set to
            # zero; otherwise, noise will leak into the feature maps.
            seqs = apply_mask(seqs, padding_mask)

        # (N, S, M) -> (N, M, S)
        seqs = seqs.transpose(1, 2)

        # This is mathematically equivalent to dot-product.
        # (N, M, S) -> (N, 2 * M, S)
        seqs = self.pointwise_conv1(seqs)

        # (N, 2 * M, S) -> (N, M, S)
        seqs = self.pointwise_conv1_activation(seqs)

        # Pad the sequence entirely on the left in case of a causal convolution.
        if self.causal_depthwise_conv:
            seqs = pad(seqs, (self.depthwise_conv.kernel_size[0] - 1, 0))

        # (N, M, S) -> (N, M, S)
        seqs = self.depthwise_conv(seqs)

        if self.batch_norm is not None:
            seqs = self.batch_norm(seqs)
        else:
            if self.layer_norm is None:
                raise InternalError("`layer_norm` is `None`.")

            # (N, M, S) -> (N, S, M)
            seqs = seqs.transpose(1, 2)

            seqs = self.layer_norm(seqs)

            # (N, S, M) -> (N, M, S)
            seqs = seqs.transpose(1, 2)

        seqs = self.depthwise_activation(seqs)

        # This is mathematically equivalent to dot-product.
        # (N, M, S) -> (N, M, S)
        seqs = self.pointwise_conv2(seqs)

        # (N, M, S) -> (N, S, M)
        seqs = seqs.transpose(1, 2)

        return seqs

    if TYPE_CHECKING:
        __call__ = forward

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"causal_depthwise_conv={self.causal_depthwise_conv}"
