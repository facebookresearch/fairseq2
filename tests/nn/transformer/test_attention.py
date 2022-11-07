# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Generator, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from fairseq2.nn.transformer.attention import default_scaled_dot_product_attention
from tests.common import TestCase
from tests.utils import tmp_rng_seed


class TestScaledDotProductAttention(TestCase):
    # TODO: Replace with `naive_scaled_dot_product_attention`.
    @staticmethod
    def _compute_attn(
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        training: bool = True,
    ) -> Tensor:
        queries = queries * (queries.size(-1) ** -0.5)

        if mask is None:
            attn_weights = torch.bmm(queries, keys.transpose(1, 2))
        else:
            attn_weights = torch.baddbmm(mask, queries, keys.transpose(1, 2))

        attn_weights = F.softmax(attn_weights, dim=-1)

        if training and dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, dropout_p, training)

        return torch.bmm(attn_weights, values)

    def _get_test_args(self) -> Generator[Dict[str, Any], None, None]:
        N = 2  # Batch
        S = 3  # Source Sequence
        T = 2  # Target Sequence
        K = 2  # Key
        V = 3  # Value

        def t(*args: int) -> Tensor:
            return torch.randn(*args, device=self.device)

        def q() -> Tensor:
            return t(N, T, K)

        def k() -> Tensor:
            return t(N, S, K)

        def v() -> Tensor:
            return t(N, S, V)

        def m() -> Tensor:
            return t(T, S)

        # fmt: off
        yield {"queries": q(), "keys": k(), "values": v(), "mask": None, "dropout_p": 0.0, "training": True}
        yield {"queries": q(), "keys": k(), "values": v(), "mask": m(),  "dropout_p": 0.0, "training": True}
        yield {"queries": q(), "keys": k(), "values": v(), "mask": None, "dropout_p": 0.5, "training": True}
        yield {"queries": q(), "keys": k(), "values": v(), "mask": m(),  "dropout_p": 0.5, "training": True}
        yield {"queries": q(), "keys": k(), "values": v(), "mask": None, "dropout_p": 0.5, "training": False}
        # fmt: on

    def test_function_computes_expected_attention(self) -> None:
        for kwargs in self._get_test_args():
            with self.subTest(**kwargs):
                with tmp_rng_seed(self.device):
                    attn, _ = default_scaled_dot_product_attention(**kwargs)

                with tmp_rng_seed(self.device):
                    expected_attn = self._compute_attn(**kwargs)

                self.assertAllClose(attn, expected_attn)
