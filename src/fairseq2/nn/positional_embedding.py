# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "PositionalEmbedding",
    "SinusoidalPositionalEmbedding",
    "LearnedPositionalEmbedding",
]

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter

from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.typing import DataType, Device


class PositionalEmbedding(Module, ABC):
    """Produces positional embeddings."""

    max_seq_len: int
    """The expected maximum sequence length."""

    embedding_dim: int
    """The dimensionality of positional embeddings."""

    def __init__(self, max_seq_len: int, embedding_dim: int) -> None:
        """
        :param max_seq_len:
            The expected maximum sequence length.
        :param embedding_dim:
            The dimensionality of positional embeddings.
        """
        super().__init__()

        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

    def forward(
        self, embed: Tensor, state_bag: Optional[IncrementalStateBag] = None
    ) -> Tensor:
        """
        :param embed:
            The token embeddings onto which the positional embeddings will be
            added. *Shape:* :math:`(N,S,E)`, or :math:`(S,E)` when unbatched,
            where :math:`N` is the batch size, :math:`S` is the sequence length,
            and :math:`E` is the embedding size.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            The token embeddings with the positional embeddings added. *Shape:*
            Same as ``embed``.
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
            The token embeddings onto which the positional embeddings will be
            added. *Shape:* :math:`(N,S,E)`, where :math:`N` is the batch size,
            :math:`S` is the sequence length, and :math:`E` is the embedding
            size.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            The token embeddings with the positional embeddings added. *Shape:*
            Same as ``embed``.

        :meta public:
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"max_seq_len={self.max_seq_len}, embedding_dim={self.embedding_dim}"


@final
class SinusoidalPositionalEmbedding(PositionalEmbedding):
    """Produces sinusoidal positional embeddings.

    The positional embeddings are initialized as in tensor2tensor which differs
    slightly from the description in section 3.5 of
    :cite:t:`DBLP:journals/corr/VaswaniSPUJGKP17`. This means instead of

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
    >>> m = SinusoidalPositionalEmbedding(max_seq_len=16, embedding_dim=4)
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
        embedding_dim: int,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__(max_seq_len, embedding_dim)

        weight = torch.empty(max_seq_len, embedding_dim, device=device, dtype=dtype)

        self.register_buffer("weight", weight, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters and buffers of the module."""
        _fill_sinusoidal(self.weight)

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
    >>> m = LearnedPositionalEmbedding(max_seq_len=16, embedding_dim=4)
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
        embedding_dim: int,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__(max_seq_len, embedding_dim)

        self.weight = Parameter(
            torch.empty(max_seq_len, embedding_dim, device=device, dtype=dtype)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters and buffers of the module."""
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

        indices = torch.arange(
            start_step, start_step + seq_len, device=embed.device, dtype=torch.int64
        )

        return embed + F.embedding(indices, self.weight)


def _fill_sinusoidal(weight: Tensor) -> None:
    num_embed, embedding_dim = weight.shape

    num_sin = embedding_dim // 2

    # Zero pad if the embedding size is odd.
    if embedding_dim > 2 * num_sin:
        weight[:, -1:] = 0

    l_half = weight[:, :num_sin]
    r_half = weight[:, num_sin:]

    fct_kwargs: Dict[str, Any] = {"device": weight.device, "dtype": weight.dtype}

    # This is identical to tensor2tensor's implementation.
    indices = torch.arange(num_embed, **fct_kwargs)

    sin = torch.exp(
        torch.arange(num_sin, **fct_kwargs) * -math.log(10000) / (num_sin - 1)
    )

    torch.matmul(indices[:, None], sin[None, :], out=l_half)

    r_half[:] = l_half[:]

    l_half.sin_()
    r_half.cos_()
