# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from torch import Tensor
from torch.nn import Conv2d, Module
from torch.nn.functional import relu
from torch.optim import AdamW as BaseAdamW

from fairseq2.optim import AdamW
from fairseq2.typing import DataType
from tests.common import assert_close, device, tmp_rng_seed


class AdamWTestNet(Module):
    def __init__(self, dtype: DataType) -> None:
        super().__init__()

        self.conv1 = Conv2d(8, 4, 1, device=device, dtype=dtype)
        self.conv2 = Conv2d(4, 2, 1, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(relu(self.conv1(x)))  # type: ignore[no-any-return]


class TestAdamW:
    def test_step_updates_fp32_params_correctly(self) -> None:
        net1, net2 = self.run_step(torch.float32)

        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            assert_close(p1, p2)

            assert p1.grad is not None
            assert p2.grad is not None

            assert_close(p1.grad, p2.grad)

    @pytest.mark.skipif(device.type != "cuda", reason="requires CUDA")
    def test_step_updates_fp16_params_correctly(self) -> None:
        net1, net2 = self.run_step(torch.float16)

        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            assert not torch.isnan(p1).any()
            assert not torch.isinf(p1).any()

            # Vanilla AdamW very likely underflowed; however, if not, we should
            # match.
            if not torch.isnan(p2).any() and not torch.isinf(p2).any():
                assert_close(p1, p2)

    def run_step(self, dtype: DataType) -> Tuple[Module, Module]:
        with tmp_rng_seed(device):
            net1 = AdamWTestNet(dtype)

        with tmp_rng_seed(device):
            net2 = AdamWTestNet(dtype)

        opt1 = AdamW(
            params=[  # type: ignore[arg-type]
                {"params": net1.conv1.parameters()},
                {"params": net1.conv2.parameters(), "lr": 0.002},
            ],
            lr=0.001,
            use_fp32=True,
        )
        opt2 = BaseAdamW(
            params=[  # type: ignore[arg-type]
                {"params": net2.conv1.parameters()},
                {"params": net2.conv2.parameters(), "lr": 0.002},
            ],
            lr=0.001,
        )

        x = torch.randn((2, 8, 12, 4), device=device, dtype=dtype)

        # Underflow in fp16.
        x /= 1000

        out1 = net1(x).sum()
        out2 = net2(x).sum()

        out1.backward()
        out2.backward()

        opt1.step()
        opt2.step()

        return net1, net2
