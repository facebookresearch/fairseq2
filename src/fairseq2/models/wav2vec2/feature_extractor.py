# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Sequence, Tuple, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import GELU, Conv1d, Dropout, GroupNorm, Module, Sequential
from torch.nn.functional import group_norm, layer_norm

from fairseq2.models.feature_extractor import SequenceFeatureExtractor
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.utils.grad import scale_grad
from fairseq2.typing import DataType, Device, finaloverride, override


@final
class Wav2Vec2FeatureExtractor(SequenceFeatureExtractor):
    """Extracts features from raw audio waveforms and embeds them in a latent
    space as described in Section 2 of
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    layers: Sequential
    layer_descs: List[Tuple[int, int, int]]
    num_channels: int
    grad_scale: float

    def __init__(
        self,
        layer_descs: Sequence[Tuple[int, int, int]],
        bias: bool,
        *,
        num_channels: int = 1,
        dropout_p: float = 0.0,
        layer_norm: bool = False,
        grad_scale: float = 1.0,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
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
        # The output dimensionality of the last feature extraction layer.
        feature_dim = layer_descs[-1][0]

        super().__init__(feature_dim)

        if len(layer_descs) == 0:
            raise ValueError("`layer_descs` must be non-empty.")

        self.layers = Sequential()

        self.num_channels = num_channels

        input_dim = num_channels

        for i, layer_desc in enumerate(layer_descs):
            output_dim, kernel_size, stride = layer_desc

            # If Layer Normalization is requested, apply it in all layers.
            if layer_norm:
                layer_norm_ = Float32LayerNorm(
                    output_dim, bias=True, device=device, dtype=dtype
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

        self.layer_descs = list(layer_descs)

        if grad_scale <= 0.0 or grad_scale > 1.0:
            raise ValueError(
                f"`grad_scale` must be greater than 0.0 and less than or equal to 1.0, but is {grad_scale} instead."
            )

        self.grad_scale = grad_scale

    @finaloverride
    def forward(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """See the base :meth:`SequenceFeatureExtractor.forward`.

        :param seqs:
            The input waveforms. *Shape:* :math:`(N,S)`, where :math:`N` is the
            batch size and :math:`(S)` is the sequence length.
        """
        # (N, S) -> (N, C, S)
        if self.num_channels == 1:
            seqs = seqs.unsqueeze(1)

        # (N, C, S) -> (N, E, S)
        features = self.layers(seqs)

        if self.grad_scale != 1.0:
            features = scale_grad(features, self.grad_scale)

        # (N, E, S) -> (N, S, E)
        features = features.transpose(1, 2)

        # Since we contracted the temporal dimension, we should re-compute
        # the sequence lengths.
        if padding_mask is not None:
            seq_lens = self._contract_seq_lens(padding_mask.seq_lens)

            padding_mask = PaddingMask(seq_lens, batch_seq_len=features.size(1))

        return features, padding_mask

    def _contract_seq_lens(self, num_frames: Tensor) -> Tensor:
        seq_lens = num_frames.clone()

        for desc in self.layer_descs:
            kernel_size, stride = desc[1], desc[2]

            seq_lens = (((seq_lens - kernel_size) / stride) + 1.0).floor()

        return seq_lens.type_as(num_frames)

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, grad_scale={self.grad_scale}"


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
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        *,
        dropout_p: float = 0.0,
        group_norm: Optional[GroupNorm] = None,
        layer_norm: Optional[LayerNorm] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__()

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
    """Represents the convolution used in
    :class:`Wav2Vec2FeatureExtractionLayer`."""

    @override
    def reset_parameters(self) -> None:
        if self.bias is not None:
            # Call the base since we want to initialize bias as in `Conv1d`.
            super().reset_parameters()

        nn.init.kaiming_normal_(self.weight)


# TODO: Move this to data pre-processing! It isn't a real feature extractor.
class Wav2Vec2FbankFeatureExtractor(SequenceFeatureExtractor):
    num_fbank_channels: int
    stride: int
    sample_every_k: int

    def __init__(
        self, num_fbank_channels: int, stride: int, *, sample_every_k: int = 1
    ):
        super().__init__(feature_dim=num_fbank_channels * stride)

        self.num_fbank_channels = num_fbank_channels
        self.stride = stride
        self.sample_every_k = sample_every_k

    @finaloverride
    def forward(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """See the base :meth:`SequenceFeatureExtractor.forward`.

        :param seqs:
            The input log-mel filterbanks. *Shape:* :math:`(N,S,C)`, where
            :math:`N` is the batch size, :math:`S` is the number of frames, and
            :math:`C` is the number of channels.
        """
        batch_size, num_frames, num_channels = seqs.shape

        if padding_mask is None:
            seq_lens = None
        else:
            seq_lens = padding_mask.seq_lens

        if (r := num_frames % self.stride) != 0:
            num_frames -= r

            seqs = seqs[:, :num_frames, :]

            if seq_lens is not None:
                seq_lens = seq_lens.clone()

                seq_lens[seq_lens > num_frames] = num_frames

        seqs = seqs.view(
            batch_size, num_frames // self.stride, num_channels * self.stride
        )

        if self.sample_every_k > 1:
            indices = torch.arange(0, batch_size, device=seqs.device)

            seqs = seqs[indices % self.sample_every_k != 0]

        if seq_lens is not None:
            # Since we contracted the temporal dimension, we should re-compute
            # the sequence lengths.
            seq_lens = self._contract_seq_lens(seq_lens)

            padding_mask = PaddingMask(seq_lens, batch_seq_len=seqs.size(1))

        return seqs, padding_mask

    def _contract_seq_lens(self, num_frames: Tensor) -> Tensor:
        num_frames = num_frames // self.stride

        if self.sample_every_k > 1:
            num_frames //= self.sample_every_k + 1

        return num_frames

    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"num_fbank_channels={self.num_fbank_channels}, "
            f"stride={self.stride}, "
            f"sample_every_k={self.sample_every_k}"
        )


class Float32LayerNorm(LayerNorm):
    """Applies Layer Normalization in single-precision."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        w, b = self.weight, self.bias

        fp32_x = x.float()
        fp32_w = w.float() if w is not None else None
        fp32_b = b.float() if b is not None else None

        y = layer_norm(fp32_x, self.normalized_shape, fp32_w, fp32_b, self.eps)

        return y.type_as(x)


class Float32GroupNorm(GroupNorm):
    """Applies Group Normalization in single-precision."""

    @override(check_signature=False)
    def forward(self, x: Tensor) -> Tensor:
        w, b = self.weight, self.bias

        fp32_x = x.float()
        fp32_w = w.float()
        fp32_b = b.float() if b is not None else None

        y = group_norm(fp32_x, self.num_groups, fp32_w, fp32_b, self.eps)

        return y.type_as(x)
