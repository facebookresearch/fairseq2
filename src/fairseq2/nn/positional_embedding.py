# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import dtype as DataType
from torch.nn import Module, Parameter


class PositionalEmbedding(Module, ABC):
    """Produces positional embeddings."""

    max_seq_len: int
    """The expected maximum sequence length."""

    embedding_dim: int
    """The dimensionality of returned positional embeddings."""

    padding_token_idx: Optional[int]
    """The index of the padding token. While producing positional embeddings,
    paddings in an input sequence will be skipped and their positional
    embeddings will be set to zero."""

    batch_first: bool
    """If ``True``, the first dimension of batched inputs and outputs represents
    the batch; otherwise, the sequence."""

    def __init__(
        self,
        max_seq_len: int,
        embedding_dim: int,
        padding_token_idx: Optional[int] = None,
        batch_first: bool = False,
    ) -> None:
        """
        :param max_seq_len:
            The expected maximum sequence length.
        :param embedding_dim:
            The dimensionality of returned positional embeddings.
        :param padding_token_idx:
            The index of the padding token. While producing positional
            embeddings, paddings in an input sequence will be skipped and their
            positional embeddings will be set to zero.
        :param batch_first:
            If ``True``, the first dimension of batched inputs and outputs
            represents the batch; otherwise, the sequence.
        """
        super().__init__()

        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.padding_token_idx = padding_token_idx
        self.batch_first = batch_first

    def forward(
        self,
        seq: Tensor,
        incremental_eval: bool = False,
    ) -> Tensor:
        """
        :param seq:
            The input sequences. *Shape:* :math:`(S)` when unbatched,
            :math:`(N,S)` when :attr:`batch_first` is ``True``, or :math:`(S,N)`
            when :attr:`batch_first` is ``False``, where :math:`N` is the batch
            size and :math:`S` is the sequence length.
        :param incremental_eval:
            If ``True`` and in eval mode, returns the positional embedding of
            the last step only.

        :returns:
            The positional embeddings. *Shape:* :math:`(*, E_{pos})`, where
            :math:`*` is the input shape and :math:`E_{pos}` is the positional
            embedding size.
        """
        seq_dim = seq.dim()

        if seq_dim > 2:
            raise ValueError(
                f"The number of dimensions of `seq` ({seq_dim}) must be 1 or 2."
            )

        if seq_dim > 1:
            if not self.batch_first:
                seq = seq.transpose(0, 1)
        else:
            seq = seq.unsqueeze(0)

        if (seq_len := seq.size(1)) > self.max_seq_len:
            raise ValueError(
                f"The input sequence length ({seq_len}) cannot be greater than {self.max_seq_len}."
            )

        embed = self._forward_core(seq, incremental_eval)

        if seq_dim > 1:
            if not self.batch_first:
                return embed.transpose(0, 1)
            else:
                return embed
        else:
            return embed.squeeze(0)

    @abstractmethod
    def _forward_core(self, seq: Tensor, incremental_eval: bool) -> Tensor:
        """
        :param seq:
            The input sequences. *Shape:* :math:`(N,S)`, where :math:`N` is the
            batch size and :math:`S` is the sequence length.
        :param incremental_eval:
            If ``True`` and in eval mode, an actual implementation should return
            the positional embedding of the last step only.

        :returns:
            The positional embeddings. *Shape:* :math:`(N,S,E_{pos})`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`E_{pos}` is the positional embedding size.

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
    >>> from fairseq2.modules import SinusoidalPositionalEmbedding
    >>>
    >>> m = SinusoidalPositionalEmbedding(
    ...    max_seq_len=16, embedding_dim=4, padding_token_idx=3)
    ... )
    >>>
    >>> s = torch.tensor([7, 2, 3, 11])
    >>>
    >>> m(s)
    tensor([[ 0.0000e+00,  0.0000e+00,  1.0000e+00,  1.0000e+00],  # pos 0
            [ 8.4147e-01,  1.0000e-04,  5.4030e-01,  1.0000e+00],  # pos 1
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],  # pad
            [ 9.0930e-01,  2.0000e-04, -4.1615e-01,  1.0000e+00]]) # pos 2
    """

    weight: Tensor

    def __init__(
        self,
        max_seq_len: int,
        embedding_dim: int,
        padding_token_idx: Optional[int] = None,
        batch_first: bool = False,
        device: Any = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__(max_seq_len, embedding_dim, padding_token_idx, batch_first)

        if padding_token_idx is None:
            num_embed = max_seq_len
        else:
            # Make space for the padding token's zero embedding.
            num_embed = max_seq_len + 1

        weight = torch.empty(num_embed, embedding_dim, device=device, dtype=dtype)  # type: ignore[arg-type]

        self.register_buffer("weight", weight, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters and buffers of the module."""
        if self.padding_token_idx is not None:
            # Fill the padding token's zero embedding.
            self.weight[0].fill_(0.0)

            out = self.weight[1:]
        else:
            out = self.weight

        num_sin = self.embedding_dim // 2

        # Zero pad if the embedding size is odd.
        if self.embedding_dim > 2 * num_sin:
            out[:, -1:] = 0

        l_half = out[:, :num_sin]
        r_half = out[:, num_sin:]

        fct_kwargs: Dict = {"device": out.device, "dtype": out.dtype}

        # This is identical to tensor2tensor's implementation.
        ind = torch.arange(out.size(0), **fct_kwargs)

        sin = torch.exp(
            torch.arange(num_sin, **fct_kwargs) * -math.log(10000) / (num_sin - 1)
        )

        torch.matmul(ind[:, None], sin[None, :], out=l_half)

        r_half[:] = l_half[:]

        l_half.sin_()
        r_half.cos_()

    def _forward_core(self, seq: Tensor, incremental_eval: bool) -> Tensor:  # override
        """:meta private:"""
        bsz, seq_len = seq.shape

        out_size = (bsz, -1, self.embedding_dim)

        last_step_only = not self.training and incremental_eval

        # Shortcut index selection if we don't expect to have padding in the
        # input sequence.
        if self.padding_token_idx is None:
            if last_step_only:
                start_step = seq_len - 1
            else:
                start_step = 0

            return self.weight[start_step:seq_len].clone().expand(out_size)
        else:
            ind = _make_indices_with_padding(
                seq, last_step_only, self.padding_token_idx
            )

            return self.weight.index_select(dim=0, index=ind.view(-1)).view(out_size)


@final
class LearnedPositionalEmbedding(PositionalEmbedding):
    """Learns positional embeddings.

    Usage:

    >>> import torch
    >>>
    >>> from fairseq2.modules import LearnedPositionalEmbedding
    >>>
    >>> m = LearnedPositionalEmbedding(
    ...    max_seq_len=16, embedding_dim=4, padding_token_idx=3)
    ... )
    >>>
    >>> s = torch.tensor([7, 2, 3, 11])
    >>>
    >>> m(s)
    tensor([[ 1.1135,  0.5548,  0.4293,  2.0112],                               # pos 0
            [ 0.2364,  0.6009,  3.3865, -2.4810],                               # pos 1
            [ 0.0000,  0.0000,  0.0000,  0.0000],                               # pad
            [-0.4746,  0.4544,  0.2761,  0.8828]], grad_fn=<SqueezeBackward1>)  # pos 3
    """

    weight: Parameter

    def __init__(
        self,
        max_seq_len: int,
        embedding_dim: int,
        padding_token_idx: Optional[int] = None,
        batch_first: bool = False,
        device: Any = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__(max_seq_len, embedding_dim, padding_token_idx, batch_first)

        if padding_token_idx is None:
            num_embed = max_seq_len
        else:
            # Make space for the padding token's zero embedding.
            num_embed = max_seq_len + 1

        self.weight = Parameter(
            torch.empty(num_embed, embedding_dim, device=device, dtype=dtype)  # type: ignore[arg-type]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters and buffers of the module."""
        nn.init.normal_(self.weight)

        if self.padding_token_idx is not None:
            with torch.no_grad():
                self.weight[0].fill_(0.0)

    def _forward_core(self, seq: Tensor, incremental_eval: bool) -> Tensor:  # override
        """:meta private:"""
        last_step_only = not self.training and incremental_eval

        if self.padding_token_idx is None:
            ind = _make_indices_sans_padding(seq, last_step_only)

            pad = None
        else:
            ind = _make_indices_with_padding(
                seq, last_step_only, self.padding_token_idx
            )

            pad = 0

        return F.embedding(ind, self.weight, padding_idx=pad)


def _make_indices_with_padding(
    seq: Tensor, last_step_only: bool, padding_token_idx: int
) -> Tensor:
    padding_mask = seq.ne(padding_token_idx).type(torch.int64)

    ind = padding_mask.cumsum(dim=-1)

    # Set the padding indices to zero.
    ind = ind * padding_mask

    if last_step_only:
        return ind[:, -1:]
    else:
        return ind


def _make_indices_sans_padding(seq: Tensor, last_step_only: bool) -> Tensor:
    bsz, seq_len = seq.shape

    if last_step_only:
        start_step = seq_len - 1
    else:
        start_step = 0

    ind = torch.arange(start_step, seq_len, device=seq.device, dtype=torch.int64)

    return ind.expand(bsz, -1)
