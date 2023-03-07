# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import ABC, abstractmethod
from typing import Optional, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter

from fairseq2.nn.incremental_state import IncrementalStateBag


class PositionalEmbedding(Module, ABC):
    """Produces positional embeddings."""

    max_seq_len: int
    embed_dim: int

    def __init__(self, max_seq_len: int, embed_dim: int) -> None:
        """
        :param max_seq_len:
            The expected maximum sequence length.
        :param embed_dim:
            The dimensionality of positional embeddings.
        """
        super().__init__()

        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

    def forward(
        self, embed: Tensor, state_bag: Optional[IncrementalStateBag] = None
    ) -> Tensor:
        """
        :param embed:
            The embeddings onto which the positional embeddings will be added.
            *Shape:* :math:`(N,S,E)`, or :math:`(S,E)` when unbatched, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`E` is the embedding size.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            The embeddings with added positional embeddings. *Shape:* Same as
            ``embed``.
        """
        embed_dim = embed.dim()

        if embed_dim == 2:
            embed = embed.unsqueeze(0)
        elif embed_dim != 3:
            raise ValueError(
                f"The number of dimensions of `embed` ({embed_dim}) must be 2 or 3."
            )

        if (seq_len := embed.size(1)) > self.max_seq_len:
            raise ValueError(
                f"The input sequence length ({seq_len}) cannot be greater than {self.max_seq_len}."
            )

        embed = self._do_forward(embed, state_bag)

        if embed_dim == 2:
            embed = embed.squeeze(0)

        return embed

    @abstractmethod
    def _do_forward(
        self, embed: Tensor, state_bag: Optional[IncrementalStateBag]
    ) -> Tensor:
        """
        :param embed:
            The embeddings onto which the positional embeddings will be added.
            *Shape:* :math:`(N,S,E)`, where :math:`N` is the batch size,
            :math:`S` is the sequence length, and :math:`E` is the embedding
            size.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            The embeddings with added positional embeddings. *Shape:* Same as
            ``embed``.

        :meta public:
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"max_seq_len={self.max_seq_len}, embed_dim={self.embed_dim}"


@final
class SinusoidalPositionalEmbedding(PositionalEmbedding):
    """Produces sinusoidal positional embeddings.

    The positional embeddings are initialized as in tensor2tensor which differs
    slightly from the description in section 3.5 of
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`. This means instead of

    .. math::
        PE_{(pos, 2i)}   = sin(pos/10000^{2i/d_{\\text{model}}})

        PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{\\text{model}}})

    we use

    .. math::
        PE_{(pos, i)} = sin(pos/10000^{i/d_{\\text{model}}})
            \\;\\text{for}\\;i\\;<    \\frac{d_{\\text{model}}}{2}

        PE_{(pos, i)} = cos(pos/10000^{i/d_{\\text{model}}})
            \\;\\text{for}\\;i\\;\\geq\\frac{d_{\\text{model}}}{2}

    See `here <https://github.com/tensorflow/tensor2tensor/pull/177>`_ for more
    information.

    Usage:

    >>> import torch
    >>>
    >>> from fairseq2.nn.positional_embedding import SinusoidalPositionalEmbedding
    >>>
    >>> m = SinusoidalPositionalEmbedding(max_seq_len=16, embed_dim=4)
    >>>
    >>> embed = torch.ones((3, 4))
    >>>
    >>> m(embed)
    tensor([[ 1.0000e+00,  1.0000e+00,  2.0000e+00,  2.0000e+00],  # pos 0
            [ 9.4147e-01,  2.0000e-04,  6.4030e-01,  2.0000e+00],  # pos 1
            [ 1.0930e-02,  3.0000e-04, -5.1615e-01,  2.0000e+00]]) # pos 2
    """

    weight: Tensor

    def __init__(
        self,
        max_seq_len: int,
        embed_dim: int,
        legacy_pad_token_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(max_seq_len, embed_dim)

        # This is a legacy parameter that should only be set when the embeddings
        # must be compatible with the original fairseq.
        if legacy_pad_token_idx is None:
            self._sin_offset = 0
        else:
            self._sin_offset = 1 + legacy_pad_token_idx

        weight = torch.empty((max_seq_len, embed_dim), device=device, dtype=dtype)

        self.register_buffer("weight", weight, persistent=False)

        self.reset_buffers()

    def reset_buffers(self, skip_persistent: bool = False) -> None:
        """Reset the buffers of the module.

        :param skip_persistent:
            If ``True``, do not reset persistent buffers.
        """
        num_sin = self.embed_dim // 2

        # Zero pad if the embedding size is odd.
        if self.embed_dim > 2 * num_sin:
            self.weight[:, -1:] = 0

        l_half = self.weight[:, :num_sin]
        r_half = self.weight[:, num_sin:]

        device, dtype = self.weight.device, self.weight.dtype

        start = self._sin_offset

        # This is identical to tensor2tensor's implementation.
        ind = torch.arange(start, start + self.max_seq_len, device=device, dtype=dtype)

        sin = torch.arange(num_sin, device=device, dtype=dtype)

        sin = torch.exp(sin * -math.log(10000) / (num_sin - 1))

        ind = ind.unsqueeze(1)
        sin = sin.unsqueeze(0)

        torch.matmul(ind, sin, out=l_half)

        r_half[:] = l_half[:]

        l_half.sin_()
        r_half.cos_()

    @finaloverride
    def _do_forward(
        self, embed: Tensor, state_bag: Optional[IncrementalStateBag]
    ) -> Tensor:
        """:meta private:"""
        bsz, seq_len = embed.shape[:2]

        if not self.training and state_bag is not None:
            start_step = state_bag.step
        else:
            start_step = 0

        return embed + self.weight[start_step : start_step + seq_len]


@final
class LearnedPositionalEmbedding(PositionalEmbedding):
    """Learns positional embeddings.

    Usage:

    >>> import torch
    >>>
    >>> from fairseq2.nn.positional_embedding import LearnedPositionalEmbedding
    >>>
    >>> m = LearnedPositionalEmbedding(max_seq_len=16, embed_dim=4)
    >>>
    >>> embed = torch.ones((3, 4))
    >>>
    >>> m(embed)
    tensor([[ 1.1135,  0.5548,  0.4293,  2.0112],                               # pos 0
            [ 0.2364,  0.6009,  3.3865, -2.4810],                               # pos 1
            [-0.4746,  0.4544,  0.2761,  0.8828]], grad_fn=<SqueezeBackward1>)  # pos 2
    """

    weight: Parameter

    def __init__(
        self,
        max_seq_len: int,
        embed_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(max_seq_len, embed_dim)

        self.weight = Parameter(
            torch.empty((max_seq_len, embed_dim), device=device, dtype=dtype)
        )

        self.reset_buffers()

    def reset_buffers(self, skip_persistent: bool = False) -> None:
        """Reset the buffers of the module.

        :param skip_persistent:
            If ``True``, do not reset persistent buffers.
        """
        nn.init.normal_(self.weight)

    @finaloverride
    def _do_forward(
        self, embed: Tensor, state_bag: Optional[IncrementalStateBag]
    ) -> Tensor:
        """:meta private:"""
        bsz, seq_len = embed.shape[:2]

        if not self.training and state_bag is not None:
            start_step = state_bag.step
        else:
            start_step = 0

        ind = torch.arange(
            start_step, start_step + seq_len, device=embed.device, dtype=torch.int64
        )

        return embed + F.embedding(ind, self.weight)


@final
class RotaryEmbedding(PositionalEmbedding):
    """Produces relative positional embeddings as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2104.09864`."""

    cos_weight: Tensor
    sin_weight: Tensor

    def __init__(
        self,
        max_seq_len: int,
        embed_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if embed_dim % 2 != 0:
            raise ValueError(f"`embed_dim` ({embed_dim}) must be even.")

        super().__init__(max_seq_len, embed_dim)

        cos = torch.empty((max_seq_len, embed_dim), device=device, dtype=dtype)
        sin = torch.empty((max_seq_len, embed_dim), device=device, dtype=dtype)

        self.register_buffer("cos_weight", cos, persistent=False)
        self.register_buffer("sin_weight", sin, persistent=False)

        self.reset_buffers()

    def reset_buffers(self, skip_persistent: bool = False) -> None:
        """Reset the buffers of the module.

        :param skip_persistent:
            If ``True``, do not reset persistent buffers.
        """
        device, dtype = self.sin_weight.device, self.sin_weight.dtype

        ind = torch.arange(self.embed_dim // 2, device=device, dtype=dtype)

        stp = torch.arange(self.max_seq_len, device=device, dtype=dtype)

        ind = ind.unsqueeze(0)
        stp = stp.unsqueeze(1)

        embed = torch.matmul(stp, 10000 ** (-2.0 * ind / self.embed_dim))

        cos = torch.cos(embed)
        sin = torch.sin(embed)

        self.cos_weight[:] = torch.repeat_interleave(cos, 2, dim=-1)
        self.sin_weight[:] = torch.repeat_interleave(sin, 2, dim=-1)

    @finaloverride
    def _do_forward(
        self, embed: Tensor, state_bag: Optional[IncrementalStateBag]
    ) -> Tensor:
        """:meta private:"""
        seq_len = embed.size(1)

        if not self.training and state_bag is not None:
            start_step = state_bag.step
        else:
            start_step = 0

        embed_swapped = self._swap_pairs(embed)

        cos = self.cos_weight[start_step : start_step + seq_len] * embed
        sin = self.sin_weight[start_step : start_step + seq_len] * embed_swapped

        return cos + sin

    @staticmethod
    def _swap_pairs(x: Tensor) -> Tensor:
        x1 = x[:, :, 0::2]
        x2 = x[:, :, 1::2]

        return torch.stack((-x2, x1), dim=-1).reshape(x.shape)
