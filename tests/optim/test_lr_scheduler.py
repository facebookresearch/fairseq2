# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence, Union

import pytest
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d, Module
from torch.optim import SGD

from fairseq2.optim.lr_scheduler import LRScheduler, MyleLR, NoamLR, PolynomialDecayLR


class LRSchedulerTestNet(Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = Conv2d(1, 1, 1)
        self.conv2 = Conv2d(1, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(F.relu(self.conv1(x)))  # type: ignore[no-any-return]


class TestLRSchedulers:
    def setup_method(self) -> None:
        self.base_lr1 = 0.05
        self.base_lr2 = 0.5

        self.net = LRSchedulerTestNet()
        self.opt = SGD(
            params=[  # type: ignore[arg-type]
                {"params": self.net.conv1.parameters()},
                {"params": self.net.conv2.parameters(), "lr": self.base_lr2},
            ],
            lr=self.base_lr1,
        )

    def step(self, s: LRScheduler) -> None:
        self.opt.step()

        s.step()

    def test_noam(self) -> None:
        num_warmup_steps = 100

        scheduler = NoamLR(self.opt, num_warmup_steps)

        assert scheduler.get_last_lr() == [0.0, 0.0]

        # In the first 100 steps, we expect the learning rate to linearly warmup
        # to its original value multiplied by the inverse square root of the
        # number of warmup steps.
        for _ in range(num_warmup_steps // 2):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # We are halfway through the warmup.
        factor = 0.5 * num_warmup_steps**-0.5

        assert lr1 == pytest.approx(factor * self.base_lr1)
        assert lr2 == pytest.approx(factor * self.base_lr2)

        for _ in range(num_warmup_steps // 2):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # Warmup should be complete.
        factor = num_warmup_steps**-0.5

        assert lr1 == pytest.approx(factor * self.base_lr1)
        assert lr2 == pytest.approx(factor * self.base_lr2)

        # We now expect the learning rate to decay by the inverse square root of
        # the step number.
        self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        factor = (num_warmup_steps + 1) ** -0.5

        assert lr1 == pytest.approx(factor * self.base_lr1)
        assert lr2 == pytest.approx(factor * self.base_lr2)

        for _ in range(5):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        factor = (num_warmup_steps + 6) ** -0.5

        assert lr1 == pytest.approx(factor * self.base_lr1)
        assert lr2 == pytest.approx(factor * self.base_lr2)

    def test_noam_with_zero_warmup(self) -> None:
        scheduler = NoamLR(self.opt, num_warmup_steps=0)

        # The decay should start from the base learning rate.
        assert scheduler.get_last_lr() == [self.base_lr1, self.base_lr2]

        for _ in range(5):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        assert lr1 == pytest.approx(self.base_lr1 * 5**-0.5)
        assert lr2 == pytest.approx(self.base_lr2 * 5**-0.5)

    @pytest.mark.parametrize("init_lr", [0.0, (0.0, 0.0), [0.02, 0.2]])
    def test_myle(self, init_lr: Union[float, Sequence[float]]) -> None:
        if isinstance(init_lr, float):
            init_lr1 = init_lr
            init_lr2 = init_lr
        else:
            init_lr1 = init_lr[0]
            init_lr2 = init_lr[1]

        num_warmup_steps = 100

        scheduler = MyleLR(self.opt, num_warmup_steps, init_lr)

        assert scheduler.get_last_lr() == [init_lr1, init_lr2]

        # In the first 100 steps, we expect the learning rate to linearly warmup
        # to its original value.
        for _ in range(num_warmup_steps // 2):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # We are halfway through the warmup.
        assert lr1 == pytest.approx(init_lr1 + (self.base_lr1 - init_lr1) / 2)
        assert lr2 == pytest.approx(init_lr2 + (self.base_lr2 - init_lr2) / 2)

        for _ in range(num_warmup_steps // 2):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # Warmup should be complete.
        assert lr1 == pytest.approx(self.base_lr1)
        assert lr2 == pytest.approx(self.base_lr2)

        # We now expect the learning rate to decay by the square root of the
        # number of warmup steps (a constant factor) multiplied by the inverse
        # square root of the step number.
        self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        factor = num_warmup_steps**0.5 * (num_warmup_steps + 1) ** -0.5

        assert lr1 == pytest.approx(factor * self.base_lr1)
        assert lr2 == pytest.approx(factor * self.base_lr2)

        for _ in range(5):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        factor = num_warmup_steps**0.5 * (num_warmup_steps + 6) ** -0.5

        assert lr1 == pytest.approx(factor * self.base_lr1)
        assert lr2 == pytest.approx(factor * self.base_lr2)

    def test_myle_raises_error_if_number_of_init_lrs_is_wrong(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"^The length of `init_lr` \(1\) does not match the number of parameter groups \(2\)\.$",
        ):
            MyleLR(self.opt, num_warmup_steps=10, init_lr=[0])

        with pytest.raises(
            ValueError,
            match=r"^The length of `init_lr` \(3\) does not match the number of parameter groups \(2\)\.$",
        ):
            MyleLR(self.opt, num_warmup_steps=10, init_lr=(0, 2, 3))

    def test_polynomial_decay(self) -> None:
        num_steps = 200

        num_warmup_steps = 100

        steps = num_steps - num_warmup_steps

        power = 1.5

        final_lr1 = 0.02
        final_lr2 = 0.2

        dist1 = self.base_lr1 - final_lr1
        dist2 = self.base_lr2 - final_lr2

        scheduler = PolynomialDecayLR(
            self.opt, num_steps, num_warmup_steps, power, [final_lr1, final_lr2]
        )

        assert scheduler.get_last_lr() == [0.0, 0.0]

        # In the first 100 steps, we expect the learning rate to linearly warmup
        # to its original value.
        for _ in range(num_warmup_steps // 2):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # We are halfway through the warmup.
        assert lr1 == pytest.approx(self.base_lr1 * 0.5)
        assert lr2 == pytest.approx(self.base_lr2 * 0.5)

        for _ in range(num_warmup_steps // 2):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # Warmup should be complete.
        assert lr1 == pytest.approx(self.base_lr1)
        assert lr2 == pytest.approx(self.base_lr2)

        # We now expect the learning rate to decay.
        self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        assert lr1 == pytest.approx(final_lr1 + dist1 * ((steps - 1) / steps) ** power)
        assert lr2 == pytest.approx(final_lr2 + dist2 * ((steps - 1) / steps) ** power)

        for _ in range(5):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        assert lr1 == pytest.approx(final_lr1 + dist1 * ((steps - 6) / steps) ** power)
        assert lr2 == pytest.approx(final_lr2 + dist2 * ((steps - 6) / steps) ** power)

        for _ in range(steps - 6):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # After num_steps, we expect the decay to stop.
        assert lr1 == pytest.approx(final_lr1)
        assert lr2 == pytest.approx(final_lr2)

        for _ in range(10):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        assert lr1 == pytest.approx(final_lr1)
        assert lr2 == pytest.approx(final_lr2)
