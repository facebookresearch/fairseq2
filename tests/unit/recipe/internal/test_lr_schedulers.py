# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Any

import pytest
from torch.optim import SGD

from fairseq2.error import InternalError
from fairseq2.nn import Linear
from fairseq2.recipe.config import (
    CosineAnnealingLRConfig,
    MyleLRConfig,
    NoamLRConfig,
    PolynomialDecayLRConfig,
    RegimeSection,
    TriStageLRConfig,
)
from fairseq2.recipe.internal.lr_schedulers import (
    _CosineAnnealingLRFactory,
    _MyleLRFactory,
    _NoamLRFactory,
    _PolynomialDecayLRFactory,
    _TriStageLRFactory,
)
from fairseq2.utils.validation import ValidationError


class TestCosineAnnealingLRFactory:
    def test_create_works(self) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD(proj.parameters(), lr=1.0)

        regime_section = RegimeSection()

        config = CosineAnnealingLRConfig(
            cycle_len=2,
            num_warmup_steps=100,
            cycle_mul=2.0,
            lr_mul=3.0,
            start_lr=0.5,
            final_lr=0.3,
            final_lr_scale=None,
        )

        factory = _CosineAnnealingLRFactory(optimizer, regime_section)

        scheduler = factory.create(config)

        start_lrs = [config.start_lr]
        final_lrs = [config.final_lr]

        assert scheduler.cycle_len == config.cycle_len
        assert scheduler.num_warmup_steps == config.num_warmup_steps
        assert scheduler.cycle_mul == config.cycle_mul
        assert scheduler.lr_mul == config.lr_mul
        assert scheduler.start_lrs == start_lrs
        assert scheduler.final_lrs == final_lrs

    def test_create_works_when_cycle_len_is_none(self) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD(proj.parameters(), lr=1.0)

        num_steps = 1000

        regime_section = RegimeSection(num_steps=num_steps)

        config = CosineAnnealingLRConfig(
            cycle_len=None,
            num_warmup_steps=100,
            cycle_mul=2.0,
            lr_mul=3.0,
            start_lr=0.5,
            final_lr=0.3,
            final_lr_scale=None,
        )

        factory = _CosineAnnealingLRFactory(optimizer, regime_section)

        scheduler = factory.create(config)

        cycle_len = num_steps - config.num_warmup_steps

        start_lrs = [config.start_lr]
        final_lrs = [config.final_lr]

        assert scheduler.cycle_len == cycle_len
        assert scheduler.num_warmup_steps == config.num_warmup_steps
        assert scheduler.cycle_mul == config.cycle_mul
        assert scheduler.lr_mul == config.lr_mul
        assert scheduler.start_lrs == start_lrs
        assert scheduler.final_lrs == final_lrs

    @pytest.mark.parametrize(
        "start_lrs,final_lrs", [[0.5, 0.3], [[0.5, 0.3], [0.3, 0.4]]]
    )
    def test_create_works_with_param_groups(
        self, start_lrs: float | list[float], final_lrs: float | list[float]
    ) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD([{"params": proj.weight}, {"params": proj.bias}], lr=1.0)  # type: ignore[arg-type]

        regime_section = RegimeSection()

        config = CosineAnnealingLRConfig(
            cycle_len=2,
            num_warmup_steps=100,
            cycle_mul=2.0,
            lr_mul=3.0,
            start_lr=start_lrs,
            final_lr=final_lrs,
            final_lr_scale=None,
        )

        factory = _CosineAnnealingLRFactory(optimizer, regime_section)

        scheduler = factory.create(config)

        num_param_groups = len(optimizer.param_groups)

        if isinstance(start_lrs, float):
            start_lrs = [start_lrs] * num_param_groups

        if isinstance(final_lrs, float):
            final_lrs = [final_lrs] * num_param_groups

        assert scheduler.cycle_len == config.cycle_len
        assert scheduler.num_warmup_steps == config.num_warmup_steps
        assert scheduler.cycle_mul == config.cycle_mul
        assert scheduler.lr_mul == config.lr_mul
        assert scheduler.start_lrs == start_lrs
        assert scheduler.final_lrs == final_lrs

    def test_create_works_when_final_scale_specified(self) -> None:
        lr = 1.0

        proj = Linear(10, 10, bias=True)

        optimizer = SGD(proj.parameters(), lr=lr)

        regime_section = RegimeSection()

        final_lr_scale = 0.1

        config = CosineAnnealingLRConfig(
            cycle_len=2,
            num_warmup_steps=100,
            cycle_mul=2.0,
            lr_mul=3.0,
            start_lr=0.5,
            final_lr=None,
            final_lr_scale=final_lr_scale,
        )

        factory = _CosineAnnealingLRFactory(optimizer, regime_section)

        scheduler = factory.create(config)

        start_lrs = [config.start_lr]
        final_lrs = [lr * final_lr_scale]

        assert scheduler.cycle_len == config.cycle_len
        assert scheduler.num_warmup_steps == config.num_warmup_steps
        assert scheduler.cycle_mul == config.cycle_mul
        assert scheduler.lr_mul == config.lr_mul
        assert scheduler.start_lrs == start_lrs
        assert scheduler.final_lrs == final_lrs

    @pytest.mark.parametrize(
        "start_lrs,final_lr_scales", [[0.5, 0.1], [[0.5, 0.3], [0.1, 0.2]]]
    )
    def test_create_works_with_param_groups_when_final_scale_specified(
        self, start_lrs: float | list[float], final_lr_scales: float | list[float]
    ) -> None:
        lr = 1.0

        proj = Linear(10, 10, bias=True)

        optimizer = SGD([{"params": proj.weight}, {"params": proj.bias}], lr=lr)  # type: ignore[arg-type]

        regime_section = RegimeSection(num_steps=100)

        config = CosineAnnealingLRConfig(
            cycle_len=2,
            num_warmup_steps=100,
            cycle_mul=2.0,
            lr_mul=3.0,
            start_lr=start_lrs,
            final_lr=None,
            final_lr_scale=final_lr_scales,
        )

        factory = _CosineAnnealingLRFactory(optimizer, regime_section)

        scheduler = factory.create(config)

        num_param_groups = len(optimizer.param_groups)

        if isinstance(start_lrs, float):
            start_lrs = [start_lrs] * num_param_groups

        if isinstance(final_lr_scales, float):
            final_lr_scales = [final_lr_scales] * num_param_groups

        final_lrs = [lr * scale for scale in final_lr_scales]

        assert scheduler.cycle_len == config.cycle_len
        assert scheduler.num_warmup_steps == config.num_warmup_steps
        assert scheduler.cycle_mul == config.cycle_mul
        assert scheduler.lr_mul == config.lr_mul
        assert scheduler.start_lrs == start_lrs
        assert scheduler.final_lrs == final_lrs

    def test_create_raises_error_when_cycle_len_and_num_steps_are_both_none(
        self,
    ) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD(proj.parameters(), lr=1.0)

        regime_section = RegimeSection(num_steps=None)

        config = CosineAnnealingLRConfig(cycle_len=None)

        factory = _CosineAnnealingLRFactory(optimizer, regime_section)

        with pytest.raises(
            ValidationError, match=r"^`regime.num_steps` must be specified when `lr_scheduler` is 'cosine_annealing' and `lr_scheduler\.config\.cycle_len` is not specified\.$"  # fmt: skip
        ):
            factory.create(config)

    def test_create_raises_error_when_start_lrs_do_not_match_param_groups(self) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD([{"params": proj.weight}, {"params": proj.bias}], lr=1.0)  # type: ignore[arg-type]

        regime_section = RegimeSection(num_steps=100)

        config = CosineAnnealingLRConfig(
            start_lr=[1.0, 2.0, 3.0], final_lr=[1.0, 2.0], final_lr_scale=None
        )

        factory = _CosineAnnealingLRFactory(optimizer, regime_section)

        with pytest.raises(
            ValidationError, match=r"^`lr_scheduler.config` is not valid: The length of `start_lr` must match the number of optimizer parameter groups \(2\), but is 3 instead\.$"  # fmt: skip
        ):
            factory.create(config)

    def test_create_raises_error_when_final_lrs_do_not_match_param_groups(self) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD([{"params": proj.weight}, {"params": proj.bias}], lr=1.0)  # type: ignore[arg-type]

        regime_section = RegimeSection(num_steps=100)

        config = CosineAnnealingLRConfig(
            start_lr=[1.0, 2.0], final_lr=[1.0, 2.0, 3.0, 4.0], final_lr_scale=None
        )

        factory = _CosineAnnealingLRFactory(optimizer, regime_section)

        with pytest.raises(
            ValidationError, match=r"^`lr_scheduler.config` is not valid: The length of `final_lr` must match the number of optimizer parameter groups \(2\), but is 4 instead\.$"  # fmt: skip
        ):
            factory.create(config)

    def test_create_raises_error_when_final_lr_scales_do_not_match_param_groups(
        self,
    ) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD([{"params": proj.weight}, {"params": proj.bias}], lr=1.0)  # type: ignore[arg-type]

        regime_section = RegimeSection(num_steps=100)

        config = CosineAnnealingLRConfig(
            start_lr=[1.0, 2.0], final_lr=None, final_lr_scale=[1.0, 2.0, 3.0, 4.0]
        )

        factory = _CosineAnnealingLRFactory(optimizer, regime_section)

        with pytest.raises(
            ValidationError, match=r"^`lr_scheduler.config` is not valid: The length of `final_lr_scale` must match the number of optimizer parameter groups \(2\), but is 4 instead\.$"  # fmt: skip
        ):
            factory.create(config)

    def test_create_raises_error_when_both_final_lr_and_final_lr_scale_are_specified(
        self,
    ) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD(proj.parameters(), lr=1.0)

        regime_section = RegimeSection(num_steps=100)

        config = CosineAnnealingLRConfig(start_lr=0.5, final_lr=0.3, final_lr_scale=0.4)

        factory = _CosineAnnealingLRFactory(optimizer, regime_section)

        with pytest.raises(
            InternalError, match=r"^`config.final_lr` and `config.final_lr_scale` are both specified\.$"  # fmt: skip
        ):
            factory.create(config)

    def test_create_raises_error_when_both_final_lr_and_final_lr_scale_are_none(
        self,
    ) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD(proj.parameters(), lr=1.0)

        regime_section = RegimeSection(num_steps=100)

        config = CosineAnnealingLRConfig(
            start_lr=0.5, final_lr=None, final_lr_scale=None
        )

        factory = _CosineAnnealingLRFactory(optimizer, regime_section)

        with pytest.raises(
            InternalError, match=r"^`config.final_lr` and `config.final_lr_scale` are both `None`\.$"  # fmt: skip
        ):
            factory.create(config)

    def test_create_warns_when_final_lr_is_larger_than_lr(self, caplog: Any) -> None:
        caplog.set_level(logging.INFO, logger="fairseq2")

        proj = Linear(10, 10, bias=True)

        optimizer = SGD([{"params": proj.weight}, {"params": proj.bias}], lr=2.0)  # type: ignore[arg-type]

        regime_section = RegimeSection(num_steps=100)

        config = CosineAnnealingLRConfig(
            start_lr=0.5, final_lr=None, final_lr_scale=[0.2, 1.2]
        )

        factory = _CosineAnnealingLRFactory(optimizer, regime_section)

        factory.create(config)

        assert caplog.record_tuples == [
            ("fairseq2", logging.WARNING, "The final learning rate (2.4) of optimizer parameter group 1 is greater than the learning rate (2.0). This means the learning rate will increase over the course of the training.")  # fmt: skip
        ]


class TestMyleLRFactory:
    def test_create_works(self) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD(proj.parameters(), lr=1.0)

        config = MyleLRConfig(num_warmup_steps=100, start_lr=0.5)

        factory = _MyleLRFactory(optimizer)

        scheduler = factory.create(config)

        start_lrs = [config.start_lr]

        assert scheduler.num_warmup_steps == config.num_warmup_steps
        assert scheduler.start_lrs == start_lrs

    @pytest.mark.parametrize("start_lrs", [0.5, [0.5, 0.3]])
    def test_create_works_with_param_groups(
        self, start_lrs: float | list[float]
    ) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD([{"params": proj.weight}, {"params": proj.bias}], lr=1.0)  # type: ignore[arg-type]

        config = MyleLRConfig(num_warmup_steps=100, start_lr=start_lrs)

        factory = _MyleLRFactory(optimizer)

        scheduler = factory.create(config)

        num_param_groups = len(optimizer.param_groups)

        if isinstance(start_lrs, float):
            start_lrs = [start_lrs] * num_param_groups

        assert scheduler.num_warmup_steps == config.num_warmup_steps
        assert scheduler.start_lrs == start_lrs

    def test_create_raises_error_when_start_lrs_do_not_match_param_groups(self) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD([{"params": proj.weight}, {"params": proj.bias}], lr=1.0)  # type: ignore[arg-type]

        config = MyleLRConfig(start_lr=[1.0, 2.0, 3.0])

        factory = _MyleLRFactory(optimizer)

        with pytest.raises(
            ValidationError, match=r"^`lr_scheduler.config` is not valid: The length of `start_lr` must match the number of optimizer parameter groups \(2\), but is 3 instead\.$"  # fmt: skip
        ):
            factory.create(config)


class TestNoamLRFactory:
    def test_create_works(self) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD(proj.parameters(), lr=1.0)

        config = NoamLRConfig(num_warmup_steps=100)

        factory = _NoamLRFactory(optimizer)

        scheduler = factory.create(config)

        assert scheduler.num_warmup_steps == config.num_warmup_steps


class TestPolynomialDecayLRFactory:
    @pytest.mark.parametrize("num_warmup_steps", [0, 20, 99])
    def test_create_works(self, num_warmup_steps: int) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD(proj.parameters(), lr=1.0)

        regime_section = RegimeSection(num_steps=100)

        config = PolynomialDecayLRConfig(
            num_warmup_steps=num_warmup_steps, power=2.0, start_lr=0.5, final_lr=0.3
        )

        factory = _PolynomialDecayLRFactory(optimizer, regime_section)

        scheduler = factory.create(config)

        start_lrs = [config.start_lr]
        final_lrs = [config.final_lr]

        assert scheduler.num_steps == regime_section.num_steps
        assert scheduler.num_warmup_steps == config.num_warmup_steps
        assert scheduler.power == config.power
        assert scheduler.start_lrs == start_lrs
        assert scheduler.final_lrs == final_lrs

    @pytest.mark.parametrize(
        "start_lrs,final_lrs", [[0.5, 0.3], [[0.5, 0.3], [0.3, 0.4]]]
    )
    def test_create_works_with_param_groups(
        self, start_lrs: float | list[float], final_lrs: float | list[float]
    ) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD([{"params": proj.weight}, {"params": proj.bias}], lr=1.0)  # type: ignore[arg-type]

        regime_section = RegimeSection(num_steps=100)

        config = PolynomialDecayLRConfig(start_lr=start_lrs, final_lr=final_lrs)

        factory = _PolynomialDecayLRFactory(optimizer, regime_section)

        scheduler = factory.create(config)

        num_param_groups = len(optimizer.param_groups)

        if isinstance(start_lrs, float):
            start_lrs = [start_lrs] * num_param_groups

        if isinstance(final_lrs, float):
            final_lrs = [final_lrs] * num_param_groups

        assert scheduler.num_steps == regime_section.num_steps
        assert scheduler.num_warmup_steps == config.num_warmup_steps
        assert scheduler.power == config.power
        assert scheduler.start_lrs == start_lrs
        assert scheduler.final_lrs == final_lrs

    def test_create_raises_error_when_num_steps_is_none(self) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD(proj.parameters(), lr=1.0)

        regime_section = RegimeSection(num_steps=None)

        config = PolynomialDecayLRConfig()

        factory = _PolynomialDecayLRFactory(optimizer, regime_section)

        with pytest.raises(
            ValidationError, match=r"^`regime.num_steps` must be specified when `lr_scheduler` is 'polynomial_decay'\.$"  # fmt: skip
        ):
            factory.create(config)

    @pytest.mark.parametrize("num_warmup_steps", [100, 200])
    def test_create_raises_error_when_num_warmup_steps_is_gte_num_steps(
        self, num_warmup_steps: int
    ) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD(proj.parameters(), lr=1.0)

        regime_section = RegimeSection(num_steps=100)

        config = PolynomialDecayLRConfig(
            num_warmup_steps=num_warmup_steps, start_lr=0.5, final_lr=0.3
        )

        factory = _PolynomialDecayLRFactory(optimizer, regime_section)

        with pytest.raises(
            ValidationError, match=rf"^`lr_scheduler.config` is not valid: `num_warmup_steps` must be less than `regime.warmup_steps` \(100\), but is {num_warmup_steps} instead\.$"  # fmt: skip
        ):
            factory.create(config)

    def test_create_raises_error_when_start_lrs_do_not_match_param_groups(self) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD([{"params": proj.weight}, {"params": proj.bias}], lr=1.0)  # type: ignore[arg-type]

        regime_section = RegimeSection(num_steps=100)

        config = PolynomialDecayLRConfig(start_lr=[1.0, 2.0, 3.0], final_lr=[1.0, 2.0])

        factory = _PolynomialDecayLRFactory(optimizer, regime_section)

        with pytest.raises(
            ValidationError, match=r"^`lr_scheduler.config` is not valid: The length of `start_lr` must match the number of optimizer parameter groups \(2\), but is 3 instead\.$"  # fmt: skip
        ):
            factory.create(config)

    def test_create_raises_error_when_final_lrs_do_not_match_param_groups(self) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD([{"params": proj.weight}, {"params": proj.bias}], lr=1.0)  # type: ignore[arg-type]

        regime_section = RegimeSection(num_steps=100)

        config = PolynomialDecayLRConfig(
            start_lr=[1.0, 2.0], final_lr=[1.0, 2.0, 3.0, 4.0]
        )

        factory = _PolynomialDecayLRFactory(optimizer, regime_section)

        with pytest.raises(
            ValidationError, match=r"^`lr_scheduler.config` is not valid: The length of `final_lr` must match the number of optimizer parameter groups \(2\), but is 4 instead\.$"  # fmt: skip
        ):
            factory.create(config)


class TestTriStageLRFactory:
    def test_create_works(self) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD(proj.parameters(), lr=1.0)

        num_steps = 100

        regime_section = RegimeSection(num_steps=num_steps)

        config = TriStageLRConfig(
            stage_ratio=(0.3, 0.2, 0.5), start_lr_scale=0.2, final_lr_scale=0.3
        )

        factory = _TriStageLRFactory(optimizer, regime_section)

        scheduler = factory.create(config)

        start_lr_scales = [config.start_lr_scale]
        final_lr_scales = [config.final_lr_scale]

        num_stage_steps = [int(r * num_steps) for r in config.stage_ratio]

        assert scheduler.num_steps == regime_section.num_steps
        assert scheduler.num_stage_steps == num_stage_steps
        assert scheduler.start_lr_scales == start_lr_scales
        assert scheduler.final_lr_scales == final_lr_scales

    @pytest.mark.parametrize(
        "start_scales,final_scales", [[0.5, 0.3], [[0.5, 0.3], [0.3, 0.4]]]
    )
    def test_create_works_with_param_groups(
        self, start_scales: float | list[float], final_scales: float | list[float]
    ) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD([{"params": proj.weight}, {"params": proj.bias}], lr=1.0)  # type: ignore[arg-type]

        num_steps = 100

        regime_section = RegimeSection(num_steps=num_steps)

        config = TriStageLRConfig(
            start_lr_scale=start_scales, final_lr_scale=final_scales
        )

        factory = _TriStageLRFactory(optimizer, regime_section)

        scheduler = factory.create(config)

        num_param_groups = len(optimizer.param_groups)

        if isinstance(start_scales, float):
            start_scales = [start_scales] * num_param_groups

        if isinstance(final_scales, float):
            final_scales = [final_scales] * num_param_groups

        num_stage_steps = [int(r * num_steps) for r in config.stage_ratio]

        assert scheduler.num_steps == regime_section.num_steps
        assert scheduler.num_stage_steps == num_stage_steps
        assert scheduler.start_lr_scales == start_scales
        assert scheduler.final_lr_scales == final_scales

    def test_create_raises_error_when_num_steps_is_none(self) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD(proj.parameters(), lr=1.0)

        regime_section = RegimeSection(num_steps=None)

        config = TriStageLRConfig()

        factory = _TriStageLRFactory(optimizer, regime_section)

        with pytest.raises(
            ValidationError, match=r"^`regime.num_steps` must be specified when `lr_scheduler` is 'tri_stage'\.$"  # fmt: skip
        ):
            factory.create(config)

    def test_create_raises_error_when_start_lrs_do_not_match_param_groups(self) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD([{"params": proj.weight}, {"params": proj.bias}], lr=1.0)  # type: ignore[arg-type]

        regime_section = RegimeSection(num_steps=100)

        config = TriStageLRConfig(
            start_lr_scale=[1.0, 2.0, 3.0], final_lr_scale=[1.0, 2.0]
        )

        factory = _TriStageLRFactory(optimizer, regime_section)

        with pytest.raises(
            ValidationError, match=r"^`lr_scheduler.config` is not valid: The length of `start_lr_scale` must match the number of optimizer parameter groups \(2\), but is 3 instead\.$"  # fmt: skip
        ):
            factory.create(config)

    def test_create_raises_error_when_final_lrs_do_not_match_param_groups(self) -> None:
        proj = Linear(10, 10, bias=True)

        optimizer = SGD([{"params": proj.weight}, {"params": proj.bias}], lr=1.0)  # type: ignore[arg-type]

        regime_section = RegimeSection(num_steps=100)

        config = TriStageLRConfig(
            start_lr_scale=[1.0, 2.0], final_lr_scale=[1.0, 2.0, 3.0, 4.0]
        )

        factory = _TriStageLRFactory(optimizer, regime_section)

        with pytest.raises(
            ValidationError, match=r"^`lr_scheduler.config` is not valid: The length of `final_lr_scale` must match the number of optimizer parameter groups \(2\), but is 4 instead\.$"  # fmt: skip
        ):
            factory.create(config)
