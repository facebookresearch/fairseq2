# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
from torch.nn import Linear, Module

from fairseq2.recipe.config import (
    AdafactorConfig,
    AdafactorGroupConfig,
    AdamWConfig,
    AdamWGroupConfig,
)
from fairseq2.recipe.internal.optim import _AdafactorFactory, _AdamWFactory


class FooModel(Module):
    def __init__(self) -> None:
        super().__init__()

        self.proj1 = Linear(10, 10, bias=True)
        self.proj2 = Linear(10, 10, bias=True)
        self.proj3 = Linear(10, 10, bias=True)


class TestAdamWFactory:
    @pytest.mark.parametrize(
        "impl,foreach,fused",
        [
            ["auto", None, None],
            ["foreach", True, None],
            ["fused", None, True],
            ["naive", False, None],
        ],
    )
    def test_create_works(
        self, impl: str, foreach: bool | None, fused: bool | None
    ) -> None:
        model = FooModel()

        config = AdamWConfig(
            lr=0.05,
            betas=(0.5, 0.9),
            eps=2.0,
            weight_decay=3.0,
            amsgrad=True,
            maximize=True,
            capturable=True,
            differentiable=not fused,
            impl=impl,  # type: ignore[arg-type]
        )

        factory = _AdamWFactory(model)

        optimizer = factory.create(config)

        assert len(optimizer.param_groups) == 1

        assert optimizer.param_groups[0]["lr"] == config.lr
        assert optimizer.param_groups[0]["betas"] == config.betas
        assert optimizer.param_groups[0]["eps"] == config.eps
        assert optimizer.param_groups[0]["weight_decay"] == config.weight_decay
        assert optimizer.param_groups[0]["amsgrad"] == config.amsgrad
        assert optimizer.param_groups[0]["maximize"] == config.maximize
        assert optimizer.param_groups[0]["capturable"] == config.capturable
        assert optimizer.param_groups[0]["differentiable"] == config.differentiable
        assert optimizer.param_groups[0]["foreach"] == foreach
        assert optimizer.param_groups[0]["fused"] == fused

    def test_create_works_with_groups(self) -> None:
        model = FooModel()

        group_config1 = AdamWGroupConfig(params=r"proj1\..*", lr=0.03, weight_decay=2.0)
        group_config2 = AdamWGroupConfig(params=r"proj2\..*", lr=0.06, eps=1.0)

        config = AdamWConfig(
            lr=0.05,
            betas=(0.5, 0.9),
            eps=2.0,
            weight_decay=3.0,
            amsgrad=True,
            maximize=True,
            capturable=True,
            differentiable=True,
            impl="foreach",
            groups=[group_config1, group_config2],
        )

        factory = _AdamWFactory(model)

        optimizer = factory.create(config)

        assert len(optimizer.param_groups) == 3

        assert optimizer.param_groups[0]["lr"] == group_config1.lr
        assert optimizer.param_groups[0]["betas"] == config.betas
        assert optimizer.param_groups[0]["eps"] == config.eps
        assert optimizer.param_groups[0]["weight_decay"] == group_config1.weight_decay
        assert optimizer.param_groups[0]["amsgrad"] == config.amsgrad
        assert optimizer.param_groups[0]["maximize"] == config.maximize
        assert optimizer.param_groups[0]["capturable"] == config.capturable
        assert optimizer.param_groups[0]["differentiable"] == config.differentiable
        assert optimizer.param_groups[0]["foreach"] == True
        assert optimizer.param_groups[0]["fused"] == None

        assert optimizer.param_groups[1]["lr"] == group_config2.lr
        assert optimizer.param_groups[1]["betas"] == config.betas
        assert optimizer.param_groups[1]["eps"] == group_config2.eps
        assert optimizer.param_groups[1]["weight_decay"] == config.weight_decay
        assert optimizer.param_groups[1]["amsgrad"] == config.amsgrad
        assert optimizer.param_groups[1]["maximize"] == config.maximize
        assert optimizer.param_groups[1]["capturable"] == config.capturable
        assert optimizer.param_groups[1]["differentiable"] == config.differentiable
        assert optimizer.param_groups[1]["foreach"] == True
        assert optimizer.param_groups[1]["fused"] == None

        assert optimizer.param_groups[2]["lr"] == config.lr
        assert optimizer.param_groups[2]["betas"] == config.betas
        assert optimizer.param_groups[2]["eps"] == config.eps
        assert optimizer.param_groups[2]["weight_decay"] == config.weight_decay
        assert optimizer.param_groups[2]["amsgrad"] == config.amsgrad
        assert optimizer.param_groups[2]["maximize"] == config.maximize
        assert optimizer.param_groups[2]["capturable"] == config.capturable
        assert optimizer.param_groups[2]["differentiable"] == config.differentiable
        assert optimizer.param_groups[2]["foreach"] == True
        assert optimizer.param_groups[2]["fused"] == None


class TestAdafactorFactory:
    def test_create_works(self) -> None:
        model = FooModel()

        config = AdafactorConfig(
            lr=0.05,
            beta2_decay=-2.0,
            eps=(1.0, 2.0),
            d=3.0,
            weight_decay=3.0,
            foreach=True,
            maximize=True,
        )

        factory = _AdafactorFactory(model)

        optimizer = factory.create(config)

        assert len(optimizer.param_groups) == 1

        assert optimizer.param_groups[0]["lr"] == config.lr
        assert optimizer.param_groups[0]["beta2_decay"] == config.beta2_decay
        assert optimizer.param_groups[0]["eps"] == config.eps
        assert optimizer.param_groups[0]["d"] == config.d
        assert optimizer.param_groups[0]["weight_decay"] == config.weight_decay
        assert optimizer.param_groups[0]["foreach"] == config.foreach
        assert optimizer.param_groups[0]["maximize"] == config.maximize

    def test_create_works_with_groups(self) -> None:
        model = FooModel()

        group_config1 = AdafactorGroupConfig(params=r"proj1\..*", lr=0.03, d=2.0)
        group_config2 = AdafactorGroupConfig(
            params=r"proj2\..*", lr=0.04, eps=(0.5, 1.0)
        )

        config = AdafactorConfig(
            lr=0.05,
            beta2_decay=-2.0,
            eps=(1.0, 2.0),
            d=3.0,
            weight_decay=3.0,
            foreach=True,
            maximize=True,
            groups=[group_config1, group_config2],
        )

        factory = _AdafactorFactory(model)

        optimizer = factory.create(config)

        assert len(optimizer.param_groups) == 3

        assert optimizer.param_groups[0]["lr"] == group_config1.lr
        assert optimizer.param_groups[0]["beta2_decay"] == config.beta2_decay
        assert optimizer.param_groups[0]["eps"] == config.eps
        assert optimizer.param_groups[0]["d"] == group_config1.d
        assert optimizer.param_groups[0]["weight_decay"] == config.weight_decay
        assert optimizer.param_groups[0]["foreach"] == config.foreach
        assert optimizer.param_groups[0]["maximize"] == config.maximize

        assert optimizer.param_groups[1]["lr"] == group_config2.lr
        assert optimizer.param_groups[1]["beta2_decay"] == config.beta2_decay
        assert optimizer.param_groups[1]["eps"] == group_config2.eps
        assert optimizer.param_groups[1]["d"] == config.d
        assert optimizer.param_groups[1]["weight_decay"] == config.weight_decay
        assert optimizer.param_groups[1]["foreach"] == config.foreach
        assert optimizer.param_groups[1]["maximize"] == config.maximize

        assert optimizer.param_groups[2]["lr"] == config.lr
        assert optimizer.param_groups[2]["beta2_decay"] == config.beta2_decay
        assert optimizer.param_groups[2]["eps"] == config.eps
        assert optimizer.param_groups[2]["d"] == config.d
        assert optimizer.param_groups[2]["weight_decay"] == config.weight_decay
        assert optimizer.param_groups[2]["foreach"] == config.foreach
        assert optimizer.param_groups[2]["maximize"] == config.maximize
