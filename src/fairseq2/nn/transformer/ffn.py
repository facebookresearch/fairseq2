# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional, final

from torch import Tensor
from torch.nn import Dropout, Module, ReLU, SiLU

from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer.layer_norm import (
    LayerNormFactory,
    create_standard_layer_norm,
)
from fairseq2.nn.transformer.norm_order import TransformerNormOrder
from fairseq2.typing import DataType, Device, finaloverride


class FeedForwardNetwork(Module, ABC):
    """Represents a Transformer feed-forward network."""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

    @abstractmethod
    def forward(self, seqs: Tensor) -> Tensor:
        """
        :param seqs:
            The sequences to project. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.

        :returns:
            The projected sequences. *Shape:* Same as ``seqs``.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardFeedForwardNetwork(FeedForwardNetwork):
    """Represents a Transformer feed-forward network as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    inner_proj: Linear
    inner_activation: Module
    inner_dropout: Optional[Dropout]
    inner_norm: Optional[LayerNorm]
    output_proj: Linear

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        bias: bool,
        *,
        inner_activation: Optional[Module] = None,
        inner_dropout_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        layer_norm_factory: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param inner_dim:
            The dimensionality of the inner projection layer.
        :param bias:
            If ``True``, both the inner and output projection learn an additive
            bias.
        :param inner_activation:
            The activation to apply to outputs of the inner projection layer. If
            ``None``, :func:`~torch.nn.ReLU` will be used.
        :param inner_dropout_p:
            The dropout probability on outputs of the inner projection layer.
        :param norm_order:
            The Layer Normalization order.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization module.
        """
        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self.inner_proj = Linear(model_dim, inner_dim, bias, device=device, dtype=dtype)

        if inner_activation is None:
            self.inner_activation = ReLU()
        else:
            self.inner_activation = inner_activation

        if inner_dropout_p > 0.0:
            self.inner_dropout = Dropout(inner_dropout_p)
        else:
            self.register_module("inner_dropout", None)

        if norm_order == TransformerNormOrder.PRE_WITH_NORMFORMER:
            self.inner_layer_norm = layer_norm_factory(
                inner_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("inner_layer_norm", None)

        self.output_proj = Linear(
            inner_dim, model_dim, bias, device=device, dtype=dtype
        )

    @finaloverride
    def forward(self, seqs: Tensor) -> Tensor:
        seqs = self.inner_proj(seqs)

        seqs = self.inner_activation(seqs)

        if self.inner_layer_norm is not None:
            seqs = self.inner_layer_norm(seqs)

        if self.inner_dropout is not None:
            seqs = self.inner_dropout(seqs)

        seqs = self.output_proj(seqs)

        return seqs


@final
class GLUFeedForwardNetwork(FeedForwardNetwork):
    """Represents a GLU-based Transformer feed-forward network as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2002.05202`"""

    gate_proj: Linear
    gate_activation: Module
    inner_dim_scale: float
    inner_dim_to_multiple: int
    inner_proj: Linear
    inner_dropout: Optional[Dropout]
    output_proj: Linear

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        bias: bool,
        *,
        gate_activation: Optional[Module] = None,
        inner_dim_scale: float = 2 / 3,
        inner_dim_to_multiple: int = 1,
        inner_dropout_p: float = 0.0,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param inner_dim:
            The non-scaled dimensionality of the inner projection layer.
        :param bias:
            If ``True``, all projections learn an additive bias.
        :param gate_activation:
            The activation to apply to outputs of the gate projection. If
            ``None``, :func:`~torch.nn.SiLU` will be used.
        :param inner_dim_scale:
            The scale factor for the dimensionality of the inner projection
            layer.
        :param inner_dim_to_multiple:
            The dimensionality of the inner projection layer is rounded up to
            the nearest multiple of this value.
        :param inner_dropout_p:
            The dropout probability on outputs of the inner projection layer.
        """
        super().__init__(model_dim)

        self.inner_dim_scale = inner_dim_scale

        if inner_dim_scale != 1.0:
            inner_dim = int(inner_dim * inner_dim_scale)

        self.inner_dim_to_multiple = inner_dim_to_multiple

        if inner_dim_to_multiple != 1:
            inner_dim = inner_dim_to_multiple * (
                (inner_dim + inner_dim_to_multiple - 1) // inner_dim_to_multiple
            )

        self.gate_proj = Linear(model_dim, inner_dim, bias, device=device, dtype=dtype)

        if gate_activation is None:
            self.gate_activation = SiLU()
        else:
            self.gate_activation = gate_activation

        self.inner_proj = Linear(model_dim, inner_dim, bias, device=device, dtype=dtype)

        if inner_dropout_p > 0.0:
            self.inner_dropout = Dropout(inner_dropout_p)
        else:
            self.register_module("inner_dropout", None)

        self.output_proj = Linear(
            inner_dim, model_dim, bias, device=device, dtype=dtype
        )

    @finaloverride
    def forward(self, seqs: Tensor) -> Tensor:
        gate = self.gate_proj(seqs)

        gate = self.gate_activation(gate)

        seqs = self.inner_proj(seqs)

        seqs = seqs * gate

        if self.inner_dropout is not None:
            seqs = self.inner_dropout(seqs)

        seqs = self.output_proj(seqs)

        return seqs

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return (
            f"{s}, "
            f"inner_dim_scale={self.inner_dim_scale}, "
            f"inner_dim_to_multiple={self.inner_dim_to_multiple}"
        )
