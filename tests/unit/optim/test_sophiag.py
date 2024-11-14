# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.nn import Conv2d, Module
from torch.nn.functional import relu

from fairseq2.optim import SophiaG
from fairseq2.utils.rng import temporary_manual_seed
from tests.common import assert_close, device


class SophiaGTestNet(Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = Conv2d(4, 2, 1, device=device, dtype=torch.float32)
        self.conv2 = Conv2d(2, 1, 1, device=device, dtype=torch.float32)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(relu(self.conv1(x)))  # type: ignore[no-any-return]


class TestSophiaG:
    @pytest.mark.skipif(device.type != "cpu", reason="requires CPU")
    def test_step_updates_params_correctly(self) -> None:
        net = self.run_step()
        torch.set_printoptions(precision=5)
        expected = [
            Tensor(
                [
                    [[[-0.03810]], [[-1.35257]], [[2.62534]], [[0.65126]]],
                    [[[0.43054]], [[0.28050]], [[-0.17763]], [[-0.77252]]],
                ]
            ),
            Tensor([6.13362, 0.70489]),
            Tensor([[[[12.91357]], [[2.88978]]]]),
            Tensor([96.0]),
        ]

        for p, expected_grad in zip(net.parameters(), expected):
            assert p.grad is not None
            assert_close(p.grad, expected_grad)

    def run_step(self) -> Module:
        with temporary_manual_seed(2, device):
            net = SophiaGTestNet()
            x = torch.randn((2, 4, 12, 4), device=device, dtype=torch.float32)

        optimizer = SophiaG(
            params=[  # type: ignore[arg-type]
                {"params": net.conv1.parameters()},
                {"params": net.conv2.parameters(), "lr": 0.002},
            ],
            lr=0.001,
        )

        out = net(x).sum()
        out.backward()
        optimizer.step()

        return net
