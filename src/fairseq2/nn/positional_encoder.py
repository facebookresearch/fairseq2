# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import ABC, abstractmethod
from typing import Optional, cast, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter

from fairseq2.nn.incremental_state import IncrementalStateBag


class PositionalEncoder(Module, ABC):
    """Encodes sequences with positional information."""

    model_dim: int
    max_seq_len: Optional[int]

    def __init__(self, model_dim: int, max_seq_len: Optional[int]) -> None:
        """
        :param model_dim:
            The dimensionality of the associated model.
        :param max_seq_len:
            The expected maximum sequence length.
        """
        super().__init__()

        self.model_dim = model_dim
        self.max_seq_len = max_seq_len

    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        """
        :param seqs:
            The sequences which will be encoded with positional information.
            *Shape:* :math:`(N,S,M)`, where :math:`N` is the batch size,
            :math:`S` is the sequence length, and :math:`M` is the
            dimensionality of the associated model.
        :param padding_mask:
            The float padding mask of ``seqs``. *Shape:* :math:`(N_{msk},S)`,
            where :math:`N_{msk}` is the batch size of the mask and :math:`S` is
            the sequence length. :math:`N` can be a multiple of :math:`N_{msk}`
            in which case the mask will be tiled before being applied.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            ``seqs`` with positional information encoded. *Shape:* Same as
            ``seqs``.
        """
        if self.max_seq_len is not None:
            if (seq_len := seqs.size(1)) > self.max_seq_len:
                raise ValueError(
                    f"The input sequence length must be less than or equal to the maximum sequence length ({self.max_seq_len}), but is {seq_len} instead."
                )

        return self._do_forward(seqs, padding_mask, state_bag)

    @abstractmethod
    def _do_forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        """
        :param seqs:
            The sequences which will be encoded with positional information.
            *Shape:* :math:`(N,S,M)`, where :math:`N` is the batch size,
            :math:`S` is the sequence length, and :math:`M` is the
            dimensionality of the associated model.
        :param padding_mask:
            The float padding mask of ``seqs``. *Shape:* :math:`(N_{msk},S)`,
            where :math:`N_{msk}` is the batch size of the mask and :math:`S` is
            the sequence length. If padding has to be applied, a derived class
            should use the :func:`~fairseq2.nn.utils.mask.apply_padding_mask`
            function.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            ``seqs`` with positional information encoded. *Shape:* Same as
            ``seqs``.

        :meta public:
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        s = f"model_dim={self.model_dim}"

        if self.max_seq_len is not None:
            s += f", max_seq_len={self.max_seq_len}"

        return s


@final
class SinusoidalPositionalEncoder(PositionalEncoder):
    """Encodes sequences with fixed sinusoidal positional information.

    The positional encodings are initialized as in tensor2tensor which differs
    slightly from the description in section 3.5 of
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`. This means instead of

    .. math::
        PE_{(pos, 2i)}   = \\text{sin}(pos/10000^{2i/d_{model}})

        PE_{(pos, 2i+1)} = \\text{cos}(pos/10000^{2i/d_{model}})

    we use

    .. math::
        PE_{(pos, i)} = \\text{sin}(pos/10000^{i/d_{model}})\\;\\text{for}\\;i\\;    <\\frac{d_{model}}{2}

        PE_{(pos, i)} = \\text{cos}(pos/10000^{i/d_{model}})\\;\\text{for}\\;i\\;\\geq\\frac{d_{model}}{2}

    See `here <https://github.com/tensorflow/tensor2tensor/pull/177>`_ for more
    information.

    Usage:

    >>> import torch
    >>>
    >>> from fairseq2.nn.positional_encoder import SinusoidalPositionalEncoder
    >>>
    >>> m = SinusoidalPositionalEncoder(model_dim=4, max_seq_len=16)
    >>>
    >>> seqs = torch.ones((3, 4))
    >>>
    >>> m(seqs)
    tensor([[ 1.0000e+00,  1.0000e+00,  2.0000e+00,  2.0000e+00],  # pos 0
            [ 9.4147e-01,  2.0000e-04,  6.4030e-01,  2.0000e+00],  # pos 1
            [ 1.0930e-02,  3.0000e-04, -5.1615e-01,  2.0000e+00]]) # pos 2
    """

    weight: Tensor

    def __init__(
        self,
        model_dim: int,
        max_seq_len: int,
        _legacy_pad_token_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(model_dim, max_seq_len)

        # This is a legacy parameter that should only be set when the encodings
        # must be compatible with fairseq.
        if _legacy_pad_token_idx is None:
            self._sin_offset = 0
        else:
            self._sin_offset = 1 + _legacy_pad_token_idx

        weight = torch.empty((max_seq_len, model_dim), device=device, dtype=dtype)

        self.register_buffer("weight", weight, persistent=False)

        self.reset_buffers()

    def reset_buffers(self, skip_persistent: bool = False) -> None:
        """Reset the buffers of the module.

        :param skip_persistent:
            If ``True``, does not reset persistent buffers.
        """
        num_sin = self.model_dim // 2

        # Zero pad if the dimensionality of the model is odd.
        if self.model_dim > 2 * num_sin:
            self.weight[:, -1:] = 0

        l_half = self.weight[:, :num_sin]
        r_half = self.weight[:, num_sin:]

        device, dtype = self.weight.device, self.weight.dtype

        start = self._sin_offset

        max_seq_len = cast(int, self.max_seq_len)

        # This is identical to tensor2tensor's implementation.
        indices = torch.arange(start, start + max_seq_len, device=device, dtype=dtype)

        indices = indices.unsqueeze(1)

        sin = torch.arange(num_sin, device=device, dtype=dtype)

        sin = torch.exp(sin * -math.log(10000) / (num_sin - 1))

        sin = sin.unsqueeze(0)

        torch.matmul(indices, sin, out=l_half)

        r_half[:] = l_half[:]

        l_half.sin_()
        r_half.cos_()

    @finaloverride
    def _do_forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        """:meta private:"""
        seq_len = seqs.size(1)

        if not self.training and state_bag is not None:
            start_step = state_bag.step
        else:
            start_step = 0

        return seqs + self.weight[start_step : start_step + seq_len]


@final
class LearnedPositionalEncoder(PositionalEncoder):
    """Encodes sequences with learned positional embeddings.

    Usage:

    >>> import torch
    >>>
    >>> from fairseq2.nn.positional_encoder import LearnedPositionalEncoder
    >>>
    >>> m = LearnedPositionalEncoder(model_dim=4, max_seq_len=16)
    >>>
    >>> seqs = torch.ones((3, 4))
    >>>
    >>> m(seqs)
    tensor([[ 1.1135,  0.5548,  0.4293,  2.0112],                               # pos 0
            [ 0.2364,  0.6009,  3.3865, -2.4810],                               # pos 1
            [-0.4746,  0.4544,  0.2761,  0.8828]], grad_fn=<SqueezeBackward1>)  # pos 2
    """

    weight: Parameter

    def __init__(
        self,
        model_dim: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(model_dim, max_seq_len)

        self.weight = Parameter(
            torch.empty((max_seq_len, model_dim), device=device, dtype=dtype)
        )

        self.reset_buffers()

    def reset_buffers(self, skip_persistent: bool = False) -> None:
        """Reset the buffers of the module.

        :param skip_persistent:
            If ``True``, does not reset persistent buffers.
        """
        nn.init.normal_(self.weight)

    @finaloverride
    def _do_forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        """:meta private:"""
        seq_len = seqs.size(1)

        if not self.training and state_bag is not None:
            start_step = state_bag.step
        else:
            start_step = 0

        indices = torch.arange(
            start_step, start_step + seq_len, device=seqs.device, dtype=torch.int64
        )

        return seqs + F.embedding(indices, self.weight)


@final
class RotaryEncoder(PositionalEncoder):
    """Encodes sequences with relative positional information as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2104.09864`."""

    cos_weight: Tensor
    sin_weight: Tensor

    def __init__(
        self,
        model_dim: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if model_dim % 2 != 0:
            raise ValueError(f"`model_dim` must be even, but is {model_dim} instead.")

        super().__init__(model_dim, max_seq_len)

        cos = torch.empty((max_seq_len, model_dim), device=device, dtype=dtype)
        sin = torch.empty((max_seq_len, model_dim), device=device, dtype=dtype)

        self.register_buffer("cos_weight", cos, persistent=False)
        self.register_buffer("sin_weight", sin, persistent=False)

        self.reset_buffers()

    def reset_buffers(self, skip_persistent: bool = False) -> None:
        """Reset the buffers of the module.

        :param skip_persistent:
            If ``True``, does not reset persistent buffers.
        """
        device, dtype = self.sin_weight.device, self.sin_weight.dtype

        max_seq_len = cast(int, self.max_seq_len)

        indices = torch.arange(self.model_dim // 2, device=device, dtype=dtype)

        indices = indices.unsqueeze(0)

        steps = torch.arange(max_seq_len, device=device, dtype=dtype)

        steps = steps.unsqueeze(1)

        embed = torch.matmul(steps, 10000 ** (-2.0 * indices / self.model_dim))

        cos = torch.cos(embed)
        sin = torch.sin(embed)

        self.cos_weight[:] = torch.repeat_interleave(cos, 2, dim=-1)
        self.sin_weight[:] = torch.repeat_interleave(sin, 2, dim=-1)

    @finaloverride
    def _do_forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        """:meta private:"""
        seq_len = seqs.size(1)

        if not self.training and state_bag is not None:
            start_step = state_bag.step
        else:
            start_step = 0

        seqs_swapped = self._swap_pairs(seqs)

        cos = self.cos_weight[start_step : start_step + seq_len] * seqs
        sin = self.sin_weight[start_step : start_step + seq_len] * seqs_swapped

        return cos + sin

    @staticmethod
    def _swap_pairs(seqs: Tensor) -> Tensor:
        x1 = seqs[..., 0::2]
        x2 = seqs[..., 1::2]

        return torch.stack((-x2, x1), dim=-1).reshape(seqs.shape)
