# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["Fairseq1SinusoidalPositionalEmbedding"]

from typing import Optional, final

import torch
from overrides import final as finaloverride
from torch import Tensor

from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.positional_embedding import PositionalEmbedding, _fill_sinusoidal
from fairseq2.nn.utils import nan
from fairseq2.typing import DataType, Device


@final
class Fairseq1SinusoidalPositionalEmbedding(PositionalEmbedding):
    """Produces sinusoidal positional embeddings.

    .. warning::
        DO NOT use this module in new code. Check out
        :class:`~fairseq2.nn.SinusoidalPositionalEmbedding` for its replacement.
    """

    weight: Tensor
    padding_token_idx: int

    def __init__(
        self,
        max_seq_len: int,
        embedding_dim: int,
        padding_token_idx: Optional[int] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__(max_seq_len, embedding_dim)

        if padding_token_idx is not None:
            self.padding_token_idx = padding_token_idx
        else:
            self.padding_token_idx = 0

        num_embed = max_seq_len + self.padding_token_idx + 1

        weight = torch.empty(num_embed, embedding_dim, device=device, dtype=dtype)

        self.register_buffer("weight", weight, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters and buffers of the module."""
        _fill_sinusoidal(self.weight)

        # We explicitly set all embeddings up to the padding token index to NaN
        # to avoid accidental misuse.
        self.weight[: self.padding_token_idx + 1, :] = nan

    @finaloverride
    def _do_forward(
        self, embed: Tensor, state_bag: Optional[IncrementalStateBag]
    ) -> Tensor:
        """:meta private:"""
        bsz, seq_len = embed.shape[:2]

        if not self.training and state_bag is not None:
            start_step = 1 + self.padding_token_idx + state_bag.step
        else:
            start_step = 1 + self.padding_token_idx

        return embed + self.weight[start_step : start_step + seq_len].clone()
