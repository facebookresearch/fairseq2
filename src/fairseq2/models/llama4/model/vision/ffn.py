# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq2.nn import Linear, Projection


class _FeedForward(torch.nn.Module):
    c_fc: Projection
    c_proj: Projection

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = True,
        dropout: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        act_on_output: bool = False,
        init_method: Callable[[Linear], None] | None = None,
    ):
        super().__init__()
        # layers
        self.c_fc = Linear(
            dim,
            hidden_dim,
            bias=bias,
            init_fn=init_method,
        )
        self.c_proj = Linear(
            hidden_dim,
            hidden_dim,
            bias=bias,
            init_fn=init_method,
        )
        self.non_linearity = act_layer()
        self.act_on_output = act_on_output
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden: torch.Tensor = self.c_fc(x)
        hidden = self.non_linearity(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = self.c_proj(hidden)

        if self.act_on_output:
            hidden = self.non_linearity(hidden)

        return hidden
