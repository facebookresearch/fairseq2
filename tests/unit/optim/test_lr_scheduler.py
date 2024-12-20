# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
from torch import Tensor
from torch.nn import Conv2d, Module
from torch.nn.functional import relu
from torch.optim import SGD

from fairseq2.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    MyleLR,
    NoamLR,
    PolynomialDecayLR,
    TriStageLR,
)
from fairseq2.optim.lr_scheduler.factory import (
    CosineAnnealingLRConfig,
    PolynomialDecayLRConfig,
    TriStageLRConfig,
    create_cosine_annealing_lr,
    create_polynomial_decay_lr,
    create_tri_stage_lr,
)


class LRSchedulerTestNet(Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = Conv2d(1, 1, 1)
        self.conv2 = Conv2d(1, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(relu(self.conv1(x)))  # type: ignore[no-any-return]


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

    def test_cosine(self) -> None:
        cycle_len = 80

        cycle_mul = 1.2

        num_warmup_steps = 100

        lr_mul = 0.5

        start_lr1 = 0.01
        start_lr2 = 0.1

        final_lr1 = 0.02
        final_lr2 = 0.2

        scheduler = CosineAnnealingLR(
            self.opt,
            cycle_len,
            num_warmup_steps,
            cycle_mul=cycle_mul,
            lr_mul=lr_mul,
            start_lr=[start_lr1, start_lr2],
            final_lr=[final_lr1, final_lr2],
        )

        assert scheduler.get_last_lr() == [start_lr1, start_lr2]

        # In the first 100 steps, we expect the learning rate to linearly warmup
        # to its original value.
        for _ in range(num_warmup_steps // 2):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # We are halfway through the warmup.
        assert lr1 == pytest.approx(start_lr1 + (self.base_lr1 - start_lr1) / 2)
        assert lr2 == pytest.approx(start_lr2 + (self.base_lr2 - start_lr2) / 2)

        for _ in range(num_warmup_steps // 2):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # Warmup should be complete.
        assert lr1 == pytest.approx(self.base_lr1)
        assert lr2 == pytest.approx(self.base_lr2)

        # We now expect the learning rate to decay from its base value to its
        # final value within the first cycle.
        self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        factor = (1 + math.cos(math.pi * 1 / cycle_len)) / 2

        assert lr1 == pytest.approx(final_lr1 + (self.base_lr1 - final_lr1) * factor)
        assert lr2 == pytest.approx(final_lr2 + (self.base_lr2 - final_lr2) * factor)

        for _ in range((cycle_len // 2) - 1):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # We are halfway through the decay.
        assert lr1 == pytest.approx(final_lr1 + (self.base_lr1 - final_lr1) / 2)
        assert lr2 == pytest.approx(final_lr2 + (self.base_lr2 - final_lr2) / 2)

        for _ in range((cycle_len // 2) - 1):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # At the last step in the cycle the learning rate should be equal to its
        # final value.
        assert lr1 == pytest.approx(final_lr1, rel=8e-4)
        assert lr2 == pytest.approx(final_lr2, rel=8e-4)

        # Start of the second cycle.
        self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # We expect the base values to be scaled by `lr_mul`.
        assert lr1 == pytest.approx(self.base_lr1 * lr_mul)
        assert lr2 == pytest.approx(self.base_lr2 * lr_mul)

        for _ in range(int(cycle_len * cycle_mul) - 1):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # At the last step in the cycle the learning rate should be equal to its
        # final value scaled by `lr_mul`.
        assert lr1 == pytest.approx(final_lr1 * lr_mul, rel=8e-4)
        assert lr2 == pytest.approx(final_lr2 * lr_mul, rel=8e-4)

        # Start of the third cycle.
        self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # We expect the base values to be scaled by `lr_mul`.
        assert lr1 == pytest.approx(self.base_lr1 * lr_mul * lr_mul)
        assert lr2 == pytest.approx(self.base_lr2 * lr_mul * lr_mul)

    def test_cosine_with_no_cycle_scale(self) -> None:
        cycle_len = 80

        num_warmup_steps = 100

        start_lr1 = 0.01
        start_lr2 = 0.1

        final_lr1 = 0.02
        final_lr2 = 0.2

        scheduler = CosineAnnealingLR(
            self.opt,
            cycle_len,
            num_warmup_steps,
            start_lr=[start_lr1, start_lr2],
            final_lr=[final_lr1, final_lr2],
        )

        assert scheduler.get_last_lr() == [start_lr1, start_lr2]

        # In the first 100 steps, we expect the learning rate to linearly warmup
        # to its original value.
        for _ in range(num_warmup_steps):
            self.step(scheduler)

        for _ in range(3):
            lr1, lr2 = scheduler.get_last_lr()

            # Warmup should be complete.
            assert lr1 == pytest.approx(self.base_lr1)
            assert lr2 == pytest.approx(self.base_lr2)

            # We now expect the learning rate to decay from its base value to
            # its final value within the first cycle.
            self.step(scheduler)

            lr1, lr2 = scheduler.get_last_lr()

            fct = (1 + math.cos(math.pi * 1 / cycle_len)) / 2

            assert lr1 == pytest.approx(final_lr1 + (self.base_lr1 - final_lr1) * fct)
            assert lr2 == pytest.approx(final_lr2 + (self.base_lr2 - final_lr2) * fct)

            for _ in range(cycle_len - 2):
                self.step(scheduler)

            lr1, lr2 = scheduler.get_last_lr()

            # At the last step in the cycle the learning rate should be equal to
            # its final value.
            assert lr1 == pytest.approx(final_lr1, rel=8e-4)
            assert lr2 == pytest.approx(final_lr2, rel=8e-4)

            # Start the next cycle.
            self.step(scheduler)

    @pytest.mark.parametrize("start_lr", [0.0, (0.0, 0.0), [0.02, 0.2]])
    def test_myle(self, start_lr: float | Sequence[float]) -> None:
        if isinstance(start_lr, float):
            start_lr1 = start_lr
            start_lr2 = start_lr
        else:
            start_lr1 = start_lr[0]
            start_lr2 = start_lr[1]

        num_warmup_steps = 100

        scheduler = MyleLR(self.opt, num_warmup_steps, start_lr=start_lr)

        assert scheduler.get_last_lr() == [start_lr1, start_lr2]

        # In the first 100 steps, we expect the learning rate to linearly warmup
        # to its original value.
        for _ in range(num_warmup_steps // 2):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # We are halfway through the warmup.
        assert lr1 == pytest.approx(start_lr1 + (self.base_lr1 - start_lr1) / 2)
        assert lr2 == pytest.approx(start_lr2 + (self.base_lr2 - start_lr2) / 2)

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

    def test_myle_raises_error_if_number_of_start_lrs_is_wrong(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"^The length of `start_lr` must be equal to the number of parameter groups \(2\), but is 1 instead\.$",
        ):
            MyleLR(self.opt, num_warmup_steps=10, start_lr=[0])

        with pytest.raises(
            ValueError,
            match=r"^The length of `start_lr` must be equal to the number of parameter groups \(2\), but is 3 instead\.$",
        ):
            MyleLR(self.opt, num_warmup_steps=10, start_lr=(0, 2, 3))

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

    def test_polynomial_decay(self) -> None:
        num_steps = 200

        num_warmup_steps = 100

        steps = num_steps - num_warmup_steps

        power = 1.5

        start_lr1 = 0.01
        start_lr2 = 0.1

        final_lr1 = 0.02
        final_lr2 = 0.2

        dist1 = self.base_lr1 - final_lr1
        dist2 = self.base_lr2 - final_lr2

        scheduler = PolynomialDecayLR(
            self.opt,
            num_steps,
            num_warmup_steps,
            power=power,
            start_lr=[start_lr1, start_lr2],
            final_lr=[final_lr1, final_lr2],
        )

        assert scheduler.get_last_lr() == [start_lr1, start_lr2]

        # In the first 100 steps, we expect the learning rate to linearly warmup
        # to its original value.
        for _ in range(num_warmup_steps // 2):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # We are halfway through the warmup.
        assert lr1 == pytest.approx(start_lr1 + (self.base_lr1 - start_lr1) / 2)
        assert lr2 == pytest.approx(start_lr2 + (self.base_lr2 - start_lr2) / 2)

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

        # After `num_steps`, we expect the decay to stop.
        assert lr1 == pytest.approx(final_lr1)
        assert lr2 == pytest.approx(final_lr2)

        for _ in range(10):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        assert lr1 == pytest.approx(final_lr1)
        assert lr2 == pytest.approx(final_lr2)

    def test_tristage(self) -> None:
        num_steps = 200

        stage_ratio = (0.1, 0.4, 0.5)

        num_stage1_steps = int(num_steps * stage_ratio[0])
        num_stage2_steps = int(num_steps * stage_ratio[1])
        num_stage3_steps = int(num_steps * stage_ratio[2])

        start_lr_scale1 = 0.05
        start_lr_scale2 = 0.01

        final_lr_scale1 = 0.1
        final_lr_scale2 = 0.2

        start_lr1 = self.base_lr1 * start_lr_scale1
        start_lr2 = self.base_lr2 * start_lr_scale2

        final_lr1 = self.base_lr1 * final_lr_scale1
        final_lr2 = self.base_lr2 * final_lr_scale2

        decay_factor1 = -math.log(final_lr_scale1) / num_stage3_steps
        decay_factor2 = -math.log(final_lr_scale2) / num_stage3_steps

        scheduler = TriStageLR(
            self.opt,
            num_steps=num_steps,
            stage_ratio=stage_ratio,
            start_lr_scale=[start_lr_scale1, start_lr_scale2],
            final_lr_scale=[final_lr_scale1, final_lr_scale2],
        )

        assert scheduler.get_last_lr() == [start_lr1, start_lr2]

        # In the first 20 steps, we expect the learning rate to linearly warmup
        # to its original value.
        for _ in range(num_stage1_steps // 2):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # We are halfway through the warmup.
        assert lr1 == pytest.approx(start_lr1 + (self.base_lr1 - start_lr1) / 2)
        assert lr2 == pytest.approx(start_lr2 + (self.base_lr2 - start_lr2) / 2)

        for _ in range(num_stage1_steps // 2):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # Warmup should be complete.
        assert lr1 == pytest.approx(self.base_lr1)
        assert lr2 == pytest.approx(self.base_lr2)

        # Start the second stage.
        self.step(scheduler)

        # In the second stage, we expect the learning rate to stay constant.
        assert lr1 == pytest.approx(self.base_lr1)
        assert lr2 == pytest.approx(self.base_lr2)

        for _ in range(num_stage2_steps - 1):
            self.step(scheduler)

        assert lr1 == pytest.approx(self.base_lr1)
        assert lr2 == pytest.approx(self.base_lr2)

        # Start the third stage.
        self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # In the third stage, we expect the learning rate to decay to its final
        # value.
        assert lr1 == pytest.approx(self.base_lr1 * math.exp(-decay_factor1))
        assert lr2 == pytest.approx(self.base_lr2 * math.exp(-decay_factor2))

        for _ in range((num_stage3_steps // 2) - 1):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        assert lr1 == pytest.approx(self.base_lr1 * math.exp(-decay_factor1 * (num_stage3_steps // 2)))  # fmt: skip
        assert lr2 == pytest.approx(self.base_lr2 * math.exp(-decay_factor2 * (num_stage3_steps // 2)))  # fmt: skip

        for _ in range(num_stage3_steps // 2):
            self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        assert lr1 == pytest.approx(final_lr1)
        assert lr2 == pytest.approx(final_lr2)

        # Move beyond the third stage.
        self.step(scheduler)

        lr1, lr2 = scheduler.get_last_lr()

        # We expect the learning rate to stay constant at its final value after
        # the third stage.
        assert lr1 == pytest.approx(final_lr1)
        assert lr2 == pytest.approx(final_lr2)

        for _ in range(100):
            self.step(scheduler)

        assert lr1 == pytest.approx(final_lr1)
        assert lr2 == pytest.approx(final_lr2)


class TestLRSchedulerFactory:
    # Common constants
    BASE_LR1: float = 0.05
    BASE_LR2: float = 0.5
    NUM_WARMUP_STEPS: int = 100
    START_LR: float = 0.01
    FINAL_LR: float = 0.02
    NUM_STEPS: int = 200

    # CosineAnnealingLR constants
    CYCLE_LEN: int = 80
    CYCLE_MUL: float = 1.2
    LR_MUL: float = 0.5
    FINAL_LR_SCALE: float = 0.2
    MAX_NUM_STEPS: int = 1000

    # PolynomialDecayLR constants
    POLY_POWER: float = 1.5

    # TriStageLR constants
    TRI_STAGE_RATIO: tuple[float, float, float] = (0.1, 0.4, 0.5)
    TRI_START_LR_SCALE: float = 0.05
    TRI_FINAL_LR_SCALE: float = 0.1

    def setup_method(self) -> None:
        """Set up the test environment with base learning rates and an optimizer."""
        self.net = LRSchedulerTestNet()
        self.opt = SGD(
            params=[  # type: ignore[arg-type]
                {"params": self.net.conv1.parameters()},
                {"params": self.net.conv2.parameters(), "lr": self.BASE_LR2},
            ],
            lr=self.BASE_LR1,
        )

    def test_create_cosine_annealing_lr(self) -> None:
        """Test creation of a CosineAnnealingLR with various configurations."""
        # Test with final_lr
        config = CosineAnnealingLRConfig(
            cycle_len=self.CYCLE_LEN,
            num_warmup_steps=self.NUM_WARMUP_STEPS,
            cycle_mul=self.CYCLE_MUL,
            lr_mul=self.LR_MUL,
            start_lr=self.START_LR,
            final_lr=self.FINAL_LR,
            final_lr_scale=None,
        )
        scheduler = create_cosine_annealing_lr(config, self.opt, self.MAX_NUM_STEPS)

        assert isinstance(scheduler, CosineAnnealingLR)
        assert scheduler.get_last_lr() == [self.START_LR, self.START_LR]

        # Test with final_lr_scale
        config = CosineAnnealingLRConfig(
            cycle_len=self.CYCLE_LEN,
            num_warmup_steps=self.NUM_WARMUP_STEPS,
            final_lr=None,
            final_lr_scale=self.FINAL_LR_SCALE,
        )
        scheduler = create_cosine_annealing_lr(config, self.opt, None)
        assert isinstance(scheduler, CosineAnnealingLR)

    @pytest.mark.parametrize(
        "final_lr, final_lr_scale, match_pattern",
        [
            (0.02, 0.2, "Both `final_lr` .* and `final_lr_scale` .* are set"),
            (None, None, "Either `final_lr` or `final_lr_scale` must be specified"),
        ],
    )
    def test_cosine_annealing_lr_final_lr_errors(
        self, final_lr: float | None, final_lr_scale: float | None, match_pattern: str
    ) -> None:
        """Test error scenarios for final_lr and final_lr_scale in CosineAnnealingLR."""
        config = CosineAnnealingLRConfig(
            final_lr=final_lr, final_lr_scale=final_lr_scale
        )
        with pytest.raises(ValueError, match=match_pattern):
            create_cosine_annealing_lr(config, self.opt, self.MAX_NUM_STEPS)

    def test_cosine_annealing_lr_cycle_len_error(self) -> None:
        """Test error when cycle_len is None and max_num_steps is also None."""
        with pytest.raises(ValueError, match="`cycle_len` must be specified"):
            config = CosineAnnealingLRConfig(cycle_len=None)
            create_cosine_annealing_lr(config, self.opt, None)

    def test_create_polynomial_decay_lr(self) -> None:
        """Test creation of a PolynomialDecayLR with various configurations."""
        config = PolynomialDecayLRConfig(
            num_steps=self.NUM_STEPS,
            num_warmup_steps=self.NUM_WARMUP_STEPS,
            power=self.POLY_POWER,
            start_lr=self.START_LR,
            final_lr=self.FINAL_LR,
        )
        scheduler = create_polynomial_decay_lr(config, self.opt, None)

        assert isinstance(scheduler, PolynomialDecayLR)
        assert scheduler.get_last_lr() == [self.START_LR, self.START_LR]

        # Test with num_steps=None and max_num_steps provided
        config = PolynomialDecayLRConfig(num_steps=None)
        scheduler = create_polynomial_decay_lr(config, self.opt, self.MAX_NUM_STEPS)
        assert isinstance(scheduler, PolynomialDecayLR)

        # Test error when both num_steps and max_num_steps are None
        with pytest.raises(ValueError, match="`max_num_steps` must be specified"):
            config = PolynomialDecayLRConfig(num_steps=None)
            create_polynomial_decay_lr(config, self.opt, None)

    def test_create_tri_stage_lr(self) -> None:
        """Test creation of a TriStageLR with various configurations."""
        config = TriStageLRConfig(
            num_steps=self.NUM_STEPS,
            stage_ratio=self.TRI_STAGE_RATIO,
            start_lr_scale=self.TRI_START_LR_SCALE,
            final_lr_scale=self.TRI_FINAL_LR_SCALE,
        )
        scheduler = create_tri_stage_lr(config, self.opt, None)

        expected_lr1 = self.BASE_LR1 * self.TRI_START_LR_SCALE
        expected_lr2 = self.BASE_LR2 * self.TRI_START_LR_SCALE

        assert isinstance(scheduler, TriStageLR)
        assert scheduler.get_last_lr() == [expected_lr1, expected_lr2]

        # Test with num_steps=None and max_num_steps provided
        config = TriStageLRConfig(num_steps=None)
        scheduler = create_tri_stage_lr(config, self.opt, self.MAX_NUM_STEPS)
        assert isinstance(scheduler, TriStageLR)

        # Test error when both num_steps and max_num_steps are None
        with pytest.raises(ValueError, match="`max_num_steps` must be specified"):
            config = TriStageLRConfig(num_steps=None)
            create_tri_stage_lr(config, self.opt, None)
