# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import pytest
import torch
import torch.nn.functional as F
from packaging import version
from torch import Tensor

from fairseq2.nn.transformer.attention import (
    default_scaled_dot_product_attention,
    naive_scaled_dot_product_attention,
    torch_scaled_dot_product_attention,
)
from tests.common import assert_close, device
from tests.rng import tmp_rng_seed

is_pt2_or_greater = version.parse(torch.__version__) >= version.parse("2.0.0")


class TestScaledDotProductAttention:
    # TODO: Replace with `naive_scaled_dot_product_attention`.
    @staticmethod
    def _compute_attn(
        x: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        training: bool = True,
    ) -> Tensor:
        x = x * (x.size(-1) ** -0.5)

        if mask is None:
            attn_weights = torch.bmm(x, keys.transpose(1, 2))
        else:
            attn_weights = torch.baddbmm(mask, x, keys.transpose(1, 2))

        attn_weights = F.softmax(attn_weights, dim=-1)

        if training and dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, dropout_p, training)

        return torch.bmm(attn_weights, values)

    @staticmethod
    def _get_kwargs_attn_fn(
        mask: bool, dropout_p: float, training: bool
    ) -> Dict[str, Any]:
        N = 2  # Batch
        S = 3  # Source Sequence
        T = 2  # Target Sequence
        K = 2  # Key
        V = 3  # Value

        def t(*args: int) -> Tensor:
            return torch.randn(*args, device=device)

        def q() -> Tensor:
            return t(N, T, K)

        def k() -> Tensor:
            return t(N, S, K)

        def v() -> Tensor:
            return t(N, S, V)

        def m() -> Tensor:
            return t(T, S)

        kwargs: Dict[str, Any] = {
            "x": q(),
            "keys": k(),
            "values": v(),
            "mask": m() if mask else None,
            "dropout_p": dropout_p,
            "training": training,
        }
        return kwargs

    # fmt: off
    @pytest.mark.parametrize(
        "mask,dropout_p,training",
        [
            (False, 0.0, True),
            (True,  0.0, True),
            (False, 0.5, True),
            (True,  0.5, True),
            (False, 0.5, False),
        ],
    )
    # fmt: on
    def test_naive_function_computes_expected_attention(
        self, mask: bool, dropout_p: float, training: bool
    ) -> None:
        kwargs = self._get_kwargs_attn_fn(mask, dropout_p, training)

        with tmp_rng_seed(device):
            attn, _ = naive_scaled_dot_product_attention(**kwargs)

        with tmp_rng_seed(device):
            expected_attn = self._compute_attn(**kwargs)

        assert_close(attn, expected_attn)

    # fmt: off
    @pytest.mark.skipif(not is_pt2_or_greater, reason="Requires PyTorch 2.0.0 or higher")
    @pytest.mark.parametrize(
        "mask,dropout_p,training",
        [
            (False, 0.0, True),
            (True,  0.0, True),
            (False, 0.5, True),
            (True,  0.5, True),
            (False, 0.5, False),
            (False, 0.9, False),
        ],
    )
    # fmt: on
    def test_torch_function_computes_expected_attention(
        self, mask: bool, dropout_p: float, training: bool
    ) -> None:
        kwargs = self._get_kwargs_attn_fn(mask, dropout_p, training)

        with tmp_rng_seed(device):
            attn, _ = torch_scaled_dot_product_attention(**kwargs)

        with tmp_rng_seed(device):
            expected_attn = self._compute_attn(**kwargs)

        assert_close(attn, expected_attn)

    # fmt: off
    @pytest.mark.parametrize(
        "mask,dropout_p,training",
        [
            (False, 0.0, True),
            (True,  0.0, True),
            (False, 0.5, True),
            (True,  0.5, True),
            (False, 0.5, False),
        ],
    )
    # fmt: on
    def test_default_function_computes_expected_attention(
        self, mask: bool, dropout_p: float, training: bool
    ) -> None:
        kwargs = self._get_kwargs_attn_fn(mask, dropout_p, training)

        with tmp_rng_seed(device):
            attn, _ = default_scaled_dot_product_attention(**kwargs)

        with tmp_rng_seed(device):
            expected_attn = self._compute_attn(**kwargs)

        assert_close(attn, expected_attn)
