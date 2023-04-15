# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import override
from torch import Tensor
from torch.nn import GELU, Conv1d, Dropout, GroupNorm, LayerNorm, Module, Sequential

from fairseq2.nn.utils.grad import scale_grad


class Wav2Vec2FeatureExtractor(Module):
    """Extracts features (i.e. latent representations) of raw audio inputs as
    described in Section 2 of :cite:t:`baevski2020wav2vec`."""

    layers: Sequential
    grad_scale: float

    def __init__(
        self,
        layer_descs: List[Tuple[int, int, int]],
        bias: bool = False,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
        norm_eps: float = 1e-5,
        grad_scale: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param layer_descs:
            A tuple of output dimension, kernel size, and stride length for each
            feature extraction layer.
        :param bias:
            If ``True``, convolutions in the feature extraction layers will
            learn an additive bias.
        :param dropout_p:
            The dropout probability on outputs of convolutions.
        :param use_layer_norm:
            If ``True``, applies Layer Normalization to outputs of convolutions
            after dropout.
        :param norm_eps:
            The epsilon value to add to the denominator of
            :class:`~torch.nn.LayerNorm` or :class:`~torch.nn.GroupNorm` modules
            for numerical stability.
        :param grad_scale:
            The scale factor for gradients of extracted features. Setting to a
            value less than 1.0 allows the feature extractor to learn at a lower
            rate than the rest of the model.
        """
        super().__init__()

        self.layers = Sequential()

        # We expect the input waveform to be one dimensional.
        inp_dim = 1

        for i, layer_desc in enumerate(layer_descs):
            out_dim, kernel_size, stride = layer_desc

            # If Layer Normalization is requested, apply it in all layers.
            if use_layer_norm:
                layer_norm = Float32LayerNorm(
                    out_dim, norm_eps, device=device, dtype=dtype
                )

                group_norm = None

            # Otherwise, apply Group Normalization in the first layer, and do
            # not apply any normalization in the other layers.
            elif i == 0:
                group_norm = Float32GroupNorm(
                    out_dim, out_dim, norm_eps, device=device, dtype=dtype
                )

                layer_norm = None
            else:
                group_norm = None
                layer_norm = None

            layer = Wav2Vec2FeatureExtractionLayer(
                inp_dim,
                out_dim,
                kernel_size,
                stride,
                bias,
                dropout_p,
                group_norm,
                layer_norm,
                device,
                dtype,
            )

            self.layers.append(layer)

            inp_dim = out_dim

        if grad_scale <= 0.0 or grad_scale > 1.0:
            raise ValueError(
                f"`grad_scale` must be greater than 0.0 and less than or equal to 1.0, but is {grad_scale} instead."
            )

        self.grad_scale = grad_scale

    def forward(self, waveforms: Tensor) -> Tensor:
        """
        :param waveforms:
            The raw audio inputs from which to extract features. *Shape:*
            :math:`(N,S)`, or :math:`(S)` when unbatched, where :math:`N` is the
            batch size and :math:`(S)` is the sequence length.

        :returns:
            The extracted features of ``waveforms``. *Shape:* :math:`(N,S,E)`,
            or :math:`(S,E)` when unbatched, where :math:`N` is the batch size,
            :math:`(S)` is the sequence length, and :math:`E` is the embedding
            size (i.e. the output dimensionality of the last feature extraction
            layer).
        """
        # (N, S) -> (N, C, S)
        x = waveforms.unsqueeze(-2)

        # (N, C, S) -> (N, E, S)
        x = self.layers(x)

        if self.grad_scale != 1.0:
            x = scale_grad(x, self.grad_scale)

        # (N, E, S) -> (N, S, E)
        x = x.transpose(-1, -2)

        return x

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"grad_scale={self.grad_scale}"


class Wav2Vec2FeatureExtractionLayer(Module):
    """Represents a feature extraction layer used in
    :class:`Wav2Vec2FeatureExtractor`."""

    conv: Conv1d
    dropout: Optional[Dropout]
    group_norm: Optional[GroupNorm]
    layer_norm: Optional[LayerNorm]
    activation: GELU

    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        kernel_size: int,
        stride: int,
        bias: bool = False,
        dropout_p: float = 0.0,
        group_norm: Optional[GroupNorm] = None,
        layer_norm: Optional[LayerNorm] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.conv = Wav2Vec2FeatureConv1d(
            inp_dim,
            out_dim,
            kernel_size,
            stride=stride,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

        if group_norm is not None:
            self.group_norm = group_norm
        else:
            self.register_module("group_norm", None)

        if layer_norm is not None:
            self.layer_norm = layer_norm
        else:
            self.register_module("layer_norm", None)

        self.activation = GELU()

    def forward(self, x: Tensor) -> Tensor:
        # (N, C_inp, S) -> (N, C_out, S)
        x = self.conv(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.group_norm is not None:
            x = self.group_norm(x)

        if self.layer_norm is not None:
            x = x.transpose(-1, -2)

            x = self.layer_norm(x)

            x = x.transpose(-1, -2)

        x = self.activation(x)

        return x


class Wav2Vec2FeatureConv1d(Conv1d):
    """Represents the convolution used in :class:`Wav2Vec2FeatureExtractionLayer`."""

    @override
    def reset_parameters(self) -> None:
        if self.bias is not None:
            # Call the base since we want to initialize bias as in `Conv1d`.
            super().reset_parameters()

        nn.init.kaiming_normal_(self.weight)


class Float32LayerNorm(LayerNorm):
    """Applies Layer Normalization in single-precision."""

    @override
    def forward(self, input: Tensor) -> Tensor:
        x = input

        w, b = self.weight, self.bias

        fp32_x = x.float()
        fp32_w = w.float()
        fp32_b = b.float() if b is not None else None

        y = F.layer_norm(fp32_x, self.normalized_shape, fp32_w, fp32_b, self.eps)

        return y.type_as(x)


class Float32GroupNorm(GroupNorm):
    """Applies Group Normalization in single-precision."""

    @override
    def forward(self, input: Tensor) -> Tensor:
        x = input

        w, b = self.weight, self.bias

        fp32_x = x.float()
        fp32_w = w.float()
        fp32_b = b.float() if b is not None else None

        y = F.group_norm(fp32_x, self.num_groups, fp32_w, fp32_b, self.eps)

        return y.type_as(x)
