# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import GELU, Conv1d, Dropout, GroupNorm, Module, Sequential
from torch.nn.functional import group_norm
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.feature_extractor import SequenceFeatureExtractor
from fairseq2.nn import BatchLayout, LayerNorm, StandardLayerNorm
from fairseq2.nn.utils.grad import scale_grad


@final
class Wav2Vec2FeatureExtractor(SequenceFeatureExtractor):
    """Extracts features from raw audio waveforms and embeds them in a latent
    space as described in Section 2 of
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    def __init__(
        self,
        layer_descs: Sequence[tuple[int, int, int]],
        bias: bool,
        *,
        num_channels: int = 1,
        dropout_p: float = 0.0,
        layer_norm: bool = False,
        grad_scale: float = 1.0,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param layer_descs:
            A tuple of output dimension, kernel size, and stride for each
            feature extraction layer.
        :param bias:
            If ``True``, convolutions learn an additive bias.
        :param num_channels:
            The number of input channels.
        :param dropout_p:
            The dropout probability on outputs of convolutions.
        :param layer_norm:
            If ``True``, applies Layer Normalization to outputs of convolutions
            after dropout.
        :param grad_scale:
            The scale factor for gradients of extracted features. Setting to a
            value less than 1.0 allows the feature extractor to learn at a lower
            rate than the rest of the model.
        """
        super().__init__()

        if not layer_descs:
            raise ValueError("`layer_descs` must not be empty.")

        self.layers = Sequential()

        if num_channels < 1:
            raise ValueError(
                f"`num_channels` must be greater than or equal to 1, but is {num_channels} instead."
            )

        self.num_channels = num_channels

        input_dim = num_channels

        for i, layer_desc in enumerate(layer_descs):
            output_dim, kernel_size, stride = layer_desc

            # If Layer Normalization is requested, apply it in all layers.
            if layer_norm:
                layer_norm_ = StandardLayerNorm(
                    output_dim, bias=True, cast_fp32=True, device=device, dtype=dtype
                )

                group_norm_ = None

            # Otherwise, apply Group Normalization in the first layer, and do
            # not apply any normalization in other layers.
            elif i == 0:
                group_norm_ = Float32GroupNorm(
                    output_dim, output_dim, device=device, dtype=dtype
                )

                layer_norm_ = None
            else:
                group_norm_ = None
                layer_norm_ = None

            layer = Wav2Vec2FeatureExtractionLayer(
                input_dim,
                output_dim,
                kernel_size,
                stride,
                bias,
                dropout_p=dropout_p,
                group_norm=group_norm_,
                layer_norm=layer_norm_,
                device=device,
                dtype=dtype,
            )

            self.layers.append(layer)

            input_dim = output_dim

        self.layer_descs = layer_descs

        if grad_scale <= 0.0 or grad_scale > 1.0:
            raise ValueError(
                f"`grad_scale` must be greater than 0.0 and less than or equal to 1.0, but is {grad_scale} instead."
            )

        self.grad_scale = grad_scale

    @override
    def forward(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> tuple[Tensor, BatchLayout]:
        """See the base :meth:`SequenceFeatureExtractor.forward`.

        :param seqs:
            The input waveforms. *Shape:* :math:`(N,S,C)`, where :math:`N` is
            the batch size, :math:`(S)` is the sequence length, and :math:`C`
            is the number of channels.
        """
        if seqs_layout.packed:
            raise ValueError("`seqs` must not be a packed batch.")

        if self.num_channels > 1:
            # (N, S, C) -> (N, C, S)
            seqs = seqs.transpose(1, 2)
        else:
            if seqs.ndim == 3:
                # (N, S, 1) -> (N, S)
                seqs = seqs.squeeze(2)

            # Transpose can cause a copy within the first convolution op. This
            # is much faster if the number of channels is 1.
            # (N, S) -> (N, 1, S)
            seqs = seqs.unsqueeze(1)

        # (N, C, S) -> (N, E, S)
        seqs = self.layers(seqs)

        if self.training and self.grad_scale != 1.0:
            seqs = scale_grad(seqs, self.grad_scale)

        # (N, E, S) -> (N, S, E)
        seqs = seqs.transpose(1, 2)

        if seqs_layout.padded:
            # Since we contracted the temporal dimension, we should re-compute
            # the sequence lengths.
            seq_lens = self._contract_seq_lens(seqs_layout.seq_lens)
        else:
            seq_lens = None

        seqs_layout = BatchLayout.of(seqs, seq_lens)

        return seqs, seqs_layout

    def _contract_seq_lens(self, seq_lens: Sequence[int]) -> list[int]:
        seq_lens = list(seq_lens)

        for desc in self.layer_descs:
            kernel_size, stride = desc[1], desc[2]

            for i in range(len(seq_lens)):
                seq_lens[i] = math.floor(((seq_lens[i] - kernel_size) / stride) + 1.0)

        return seq_lens

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"grad_scale={self.grad_scale:G}"


@final
class Wav2Vec2FeatureExtractionLayer(Module):
    """Represents a feature extraction layer used in
    :class:`Wav2Vec2FeatureExtractor`."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        *,
        dropout_p: float = 0.0,
        group_norm: GroupNorm | None = None,
        layer_norm: LayerNorm | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.conv: Conv1d

        self.conv = Wav2Vec2FeatureConv1d(
            input_dim,
            output_dim,
            kernel_size,
            stride=stride,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        if dropout_p > 0.0:
            dropout = Dropout(dropout_p)
        else:
            dropout = None

        self.dropout: Dropout | None

        self.register_module("dropout", dropout)

        self.group_norm: GroupNorm | None

        self.register_module("group_norm", group_norm)

        self.layer_norm: LayerNorm | None

        self.register_module("layer_norm", layer_norm)

        self.activation = GELU()

    def forward(self, seqs: Tensor) -> Tensor:
        # (N, C_inp, S) -> (N, C_out, S)
        seqs = self.conv(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        if self.group_norm is not None:
            # The padding ratio of `seqs` must be as low as possible since the
            # Group Normalization implementation in PyTorch has no support for
            # padding and a large ratio can skew normalization.
            seqs = self.group_norm(seqs)

        if self.layer_norm is not None:
            seqs = seqs.transpose(1, 2)

            seqs = self.layer_norm(seqs)

            seqs = seqs.transpose(1, 2)

        seqs = self.activation(seqs)

        return seqs

    if TYPE_CHECKING:
        __call__ = forward


@final
class Wav2Vec2FeatureConv1d(Conv1d):
    """Represents the convolution used in
    :class:`Wav2Vec2FeatureExtractionLayer`."""

    @override
    def reset_parameters(self) -> None:
        if self.bias is not None:
            # Call the base since we want to initialize bias as in `Conv1d`.
            super().reset_parameters()

        nn.init.kaiming_normal_(self.weight)


# TODO: Move this to data pre-processing! It isn't a real feature extractor.
@final
class Wav2Vec2FbankFeatureExtractor(SequenceFeatureExtractor):
    def __init__(
        self, num_fbank_channels: int, stride: int, *, sample_every_k: int = 1
    ):
        super().__init__()

        self.num_fbank_channels = num_fbank_channels
        self.stride = stride
        self.sample_every_k = sample_every_k

    @override
    def forward(
        self, seqs: Tensor, seqs_layout: BatchLayout
    ) -> tuple[Tensor, BatchLayout]:
        """See the base :meth:`SequenceFeatureExtractor.forward`.

        :param seqs:
            The input log-mel filterbanks. *Shape:* :math:`(N,S,C)`, where
            :math:`N` is the batch size, :math:`S` is the number of frames, and
            :math:`C` is the number of channels.
        """
        if seqs_layout.packed:
            raise ValueError("`seqs` must not be a packed batch.")

        batch_size, num_frames, num_channels = seqs.shape

        r = num_frames % self.stride
        if r != 0:
            num_frames -= r

            seqs = seqs[:, :num_frames, :]

        seqs = seqs.view(
            batch_size, num_frames // self.stride, num_channels * self.stride
        )

        if self.sample_every_k > 1:
            indices = torch.arange(0, batch_size, device=seqs.device)

            seqs = seqs[indices % self.sample_every_k != 0]

        if seqs_layout.padded:
            # Since we contracted the temporal dimension, we should re-compute
            # the sequence lengths.
            seq_lens = self._contract_seq_lens(seqs_layout.seq_lens, num_frames)
        else:
            seq_lens = None

        seqs_layout = BatchLayout.of(seqs, seq_lens)

        return seqs, seqs_layout

    def _contract_seq_lens(
        self, seq_lens: Sequence[int], batch_width: int
    ) -> list[int]:
        seq_lens = list(seq_lens)

        for i in range(len(seq_lens)):
            seq_lens[i] = min(seq_lens[i], batch_width) // self.stride

            if self.sample_every_k > 1:
                seq_lens[i] //= self.sample_every_k + 1

        return seq_lens

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"num_fbank_channels={self.num_fbank_channels}, "
            f"stride={self.stride}, "
            f"sample_every_k={self.sample_every_k}"
        )


@final
class Float32GroupNorm(GroupNorm):
    """Applies Group Normalization in single-precision."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        w, b = self.weight, self.bias

        fp32_x = x.float()
        fp32_w = w.float() if w is not None else None  # type: ignore[redundant-expr]
        fp32_b = b.float() if b is not None else None  # type: ignore[redundant-expr]

        y = group_norm(fp32_x, self.num_groups, fp32_w, fp32_b, self.eps)

        return y.type_as(x)
