# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Final

import pytest
import torch
from packaging import version
from torch import Tensor

from fairseq2.nn.transformer.attention import NaiveSDPA, TorchSDPA
from tests.common import assert_close, device
from tests.rng import tmp_rng_seed

IS_PT2_OR_GREATER: Final = version.parse(torch.__version__) >= version.parse("2.0.0")


class TestScaledDotProductAttention:
    # fmt: off
    @pytest.mark.skipif(not IS_PT2_OR_GREATER, reason="requires PyTorch 2.0.0 or greater")
    @pytest.mark.parametrize("mask,attn_dropout_p,training",
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
    def test_torch_sdpa(
        self, mask: bool, attn_dropout_p: float, training: bool
    ) -> None:
        torch_sdpa = TorchSDPA(attn_dropout_p)
        naive_sdpa = NaiveSDPA(attn_dropout_p)

        if training:
            torch_sdpa.eval()
            naive_sdpa.eval()

        attn_args = self._get_attn_args(mask)

        with tmp_rng_seed(device):
            attn1, _ = torch_sdpa(**attn_args)

        with tmp_rng_seed(device):
            attn2, _ = naive_sdpa(**attn_args)

        assert_close(attn1, attn2)

    @staticmethod
    def _get_attn_args(mask: bool) -> Dict[str, Any]:
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
            "queries": q(),
            "keys": k(),
            "values": v(),
            "mask": m() if mask else None,
        }

        return kwargs
