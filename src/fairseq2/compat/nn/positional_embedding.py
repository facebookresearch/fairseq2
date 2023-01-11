# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["FairseqSinusoidalPositionalEmbedding"]

from typing import Optional, final

import torch
from overrides import final as finaloverride
from torch import Tensor

from fairseq2.nn.positional_embedding import (
    PositionalEmbedding,
    _fill_sinusoidal,
    _make_indices_with_padding,
)
from fairseq2.typing import DataType, Device


@final
class FairseqSinusoidalPositionalEmbedding(PositionalEmbedding):
    """Produces sinusoidal positional embeddings compatible with the original
    fairseq.

    .. warning::
        This class SHOULD NOT be used for any new code as it produces incorrect
        embeddings. It is made available only to load original fairseq models in
        fairseq2. Check out :class:`~fairseq2.nn.SinusoidalPositionalEmbedding`
        for its replacement.
    """

    weight: Tensor

    def __init__(
        self,
        max_seq_len: int,
        embedding_dim: int,
        padding_token_idx: Optional[int] = None,
        batch_first: bool = False,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__(max_seq_len, embedding_dim, padding_token_idx, batch_first)

        # Fairseq always allocates an embedding for the padding token, even if
        # not requested.
        if padding_token_idx is None:
            num_embed = max_seq_len + 1
        else:
            num_embed = max_seq_len + 1 + padding_token_idx

        weight = torch.empty(num_embed, embedding_dim, device=device, dtype=dtype)

        self.register_buffer("weight", weight, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters and buffers of the module."""
        _fill_sinusoidal(self.weight)

        padding_token_idx = self.padding_token_idx or 0

        # Fairseq treats the positional embedding at padding_token_idx as the
        # padding token's zero embedding.
        self.weight[padding_token_idx, :] = 0

        # We explicitly set all embeddings before padding_token_idx to NaN to
        # notice accidental misuse.
        self.weight[:padding_token_idx, :] = float("nan")

    @finaloverride
    def _forward_core(self, seq: Tensor, incremental_eval: bool) -> Tensor:
        """:meta private:"""
        bsz, seq_len = seq.shape

        out_size = (bsz, -1, self.embedding_dim)

        last_step_only = not self.training and incremental_eval

        padding_token_idx = self.padding_token_idx or 0

        ind = _make_indices_with_padding(seq, last_step_only, padding_token_idx)

        # Prevent the use of low frequency embeddings.
        ind += padding_token_idx

        return self.weight.index_select(dim=0, index=ind.view(-1)).view(out_size)
