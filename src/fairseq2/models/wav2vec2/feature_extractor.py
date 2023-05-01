# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Sequence, Tuple, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import final as finaloverride
from overrides import override
from torch import Tensor
from torch.nn import GELU, Conv1d, Dropout, GroupNorm, LayerNorm, Module, Sequential

from fairseq2.models.feature_extractor import FeatureExtractor
from fairseq2.nn.utils.grad import scale_grad


@final
class Wav2Vec2FeatureExtractor(FeatureExtractor):
    """Extracts features from raw audio waveforms and embeds them in a latent
    space as described in Section 2 of :cite:t:`baevski2020wav2vec`.

    .. note::
        The input waveforms are expected to be of shape :math:`(N,S)`, where
        :math:`N` is the batch size and :math:`(S)` is the sequence length.
    """

    layers: Sequential
    layer_descs: List[Tuple[int, int, int]]
    grad_scale: float

    def __init__(
        self,
        layer_descs: Sequence[Tuple[int, int, int]],
        bias: bool = False,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
        grad_scale: float = 1.0,
        norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param layer_descs:
            A tuple of output dimension, kernel size, and stride length for each
            feature extraction layer.
        :param bias:
            If ``True``, convolutions in the feature extraction layers learn an
            additive bias.
        :param dropout_p:
            The dropout probability on outputs of convolutions.
        :param use_layer_norm:
            If ``True``, applies Layer Normalization to outputs of convolutions
            after dropout.
        :param grad_scale:
            The scale factor for gradients of extracted features. Setting to a
            value less than 1.0 allows the feature extractor to learn at a lower
            rate than the rest of the model.
        :param norm_eps:
            The epsilon value to add to the denominator of
            :class:`~torch.nn.LayerNorm` or :class:`~torch.nn.GroupNorm` modules
            for numerical stability.
        """
        # The output dimensionality of the last feature extraction layer.
        embed_dim = layer_descs[-1][0]

        super().__init__(embed_dim)

        if not layer_descs:
            raise ValueError("`layer_descs` must be non-empty.")

        self.layers = Sequential()

        # We expect the input waveforms to be one dimensional.
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

        self.layer_descs = list(layer_descs)

        if grad_scale <= 0.0 or grad_scale > 1.0:
            raise ValueError(
                f"`grad_scale` must be greater than 0.0 and less than or equal to 1.0, but is {grad_scale} instead."
            )

        self.grad_scale = grad_scale

    @finaloverride
    def forward(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # (N, S) -> (N, C, S)
        seqs = seqs.unsqueeze(1)

        # (N, C, S) -> (N, E, S)
        seqs = self.layers(seqs)

        if self.grad_scale != 1.0:
            seqs = scale_grad(seqs, self.grad_scale)

        # (N, E, S) -> (N, S, E)
        seqs = seqs.transpose(1, 2)

        if seq_lens is not None:
            # Since we contracted the temporal dimension, we should re-compute
            # the sequence lengths.
            seq_lens = self._compute_seq_lens(seq_lens)

        return seqs, seq_lens

    def _compute_seq_lens(self, num_frames: Tensor) -> Tensor:
        seq_lens = num_frames.clone()

        for desc in self.layer_descs:
            kernel_size, stride = desc[1], desc[2]

            seq_lens = (((seq_lens - kernel_size) / stride) + 1.0).floor()

        return seq_lens.type(num_frames.dtype)

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return s + f", grad_scale={self.grad_scale}"


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

    def forward(self, seqs: Tensor) -> Tensor:
        # (N, C_inp, S) -> (N, C_out, S)
        seqs = self.conv(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        if self.group_norm is not None:
            seqs = self.group_norm(seqs)

        if self.layer_norm is not None:
            seqs = seqs.transpose(1, 2)

            seqs = self.layer_norm(seqs)

            seqs = seqs.transpose(1, 2)

        seqs = self.activation(seqs)

        return seqs


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
