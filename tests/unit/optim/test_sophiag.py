# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch
from torch import Tensor, tensor
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

        weights = [
            [
                [[[0.11474]], [[-0.11898]], [[0.13711]], [[-0.02554]]],
                [[[0.21359]], [[0.11903]], [[-0.05746]], [[-0.40423]]],
            ],
            [0.11416, -0.44267],
            [[[[0.09293]], [[0.04699]]]],
            [-0.15549],
        ]

        expected = list(map(lambda t: tensor(t, device=device), weights))

        for p, weight in zip(net.parameters(), expected):
            assert_close(p.data, weight)

    def run_step(self) -> Module:
        with temporary_manual_seed(2, device):
            net = SophiaGTestNet()
            x = torch.randn((2, 4, 12, 4), device=device, dtype=torch.float32)

        optimizer = SophiaG(
            params=[  # type: ignore[arg-type]
                {"params": net.conv1.parameters()},
                {"params": net.conv2.parameters()},
            ],
        )

        out = net(x).sum()
        out.backward()
        optimizer.step()

        return net
