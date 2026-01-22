# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import chain
from typing import Any

from torch.nn import Linear, Module

from fairseq2.recipe.config import Default, ParameterGroupConfig, default
from fairseq2.recipe.optim import prepare_parameter_groups


class FooModel(Module):
    def __init__(self) -> None:
        super().__init__()

        self.proj1 = Linear(10, 10, bias=True)
        self.proj2 = Linear(10, 10, bias=True)
        self.proj3 = Linear(10, 10, bias=True)


@dataclass(kw_only=True)
class FooParamGroupConfig(ParameterGroupConfig):
    lr: float | Default = default
    betas: tuple[float, float] | Default = default
    weight_decay: float | Default = default


class TestPrepareParameterGroups:
    def test_works(self, caplog: Any) -> None:
        caplog.set_level(logging.INFO, logger="fairseq2")

        model = FooModel()

        configs = [
            FooParamGroupConfig(params=r"proj1\..*", lr=0.1),
            FooParamGroupConfig(
                params=[r"proj2\.weight", r"proj2\.bias"], lr=0.2, weight_decay=0.3
            ),
        ]

        parameters = list(prepare_parameter_groups(model, configs))

        assert len(parameters) == 3

        assert isinstance(parameters[0], dict)
        assert isinstance(parameters[1], dict)
        assert isinstance(parameters[2], dict)

        assert len(parameters[0]) == 2
        assert len(parameters[1]) == 3
        assert len(parameters[2]) == 1

        assert parameters[0]["params"] == list(model.proj1.parameters())  # type: ignore[union-attr]
        assert parameters[1]["params"] == list(model.proj2.parameters())  # type: ignore[union-attr]
        assert parameters[2]["params"] == list(model.proj3.parameters())  # type: ignore[union-attr]

        assert parameters[0]["lr"] == 0.1
        assert parameters[1]["lr"] == 0.2

        assert "lr" not in parameters[2]

        assert parameters[1]["weight_decay"] == 0.3

        assert "weight_decay" not in parameters[0]
        assert "weight_decay" not in parameters[2]

        assert "betas" not in parameters[0]
        assert "betas" not in parameters[1]
        assert "betas" not in parameters[2]

        assert len(caplog.records) == 3

        assert caplog.record_tuples[0] == ("fairseq2", logging.INFO, "Optimizer Parameter Group 0: proj1.bias, proj1.weight")  # fmt: skip
        assert caplog.record_tuples[1] == ("fairseq2", logging.INFO, "Optimizer Parameter Group 1: proj2.bias, proj2.weight")  # fmt: skip
        assert caplog.record_tuples[2] == ("fairseq2", logging.INFO, "Optimizer Parameter Group 2: proj3.bias, proj3.weight")  # fmt: skip

    def test_works_when_no_groups_specified(self, caplog: Any) -> None:
        caplog.set_level(logging.INFO, logger="fairseq2")

        model = FooModel()

        parameters = prepare_parameter_groups(model, [])

        assert list(parameters) == list(model.parameters())

        assert len(caplog.records) == 0

    def test_works_when_groups_are_exhaustive(self, caplog: Any) -> None:
        caplog.set_level(logging.INFO, logger="fairseq2")

        model = FooModel()

        configs = [
            FooParamGroupConfig(params=r"proj1\..*", lr=0.1),
            FooParamGroupConfig(params=r"proj.*", lr=0.2, weight_decay=0.3),
        ]

        parameters = list(prepare_parameter_groups(model, configs))

        assert len(parameters) == 2

        assert isinstance(parameters[0], dict)
        assert isinstance(parameters[1], dict)

        assert len(parameters[0]) == 2
        assert len(parameters[1]) == 3

        p = chain(
            model.proj2.parameters(),  # type: ignore[union-attr]
            model.proj3.parameters(),  # type: ignore[union-attr]
        )

        assert parameters[0]["params"] == list(model.proj1.parameters())  # type: ignore[union-attr]
        assert parameters[1]["params"] == list(p)

        assert parameters[0]["lr"] == 0.1
        assert parameters[1]["lr"] == 0.2

        assert parameters[1]["weight_decay"] == 0.3

        assert "weight_decay" not in parameters[0]

        assert "betas" not in parameters[0]
        assert "betas" not in parameters[1]

        assert len(caplog.records) == 2

        assert caplog.record_tuples[0] == ("fairseq2", logging.INFO, "Optimizer Parameter Group 0: proj1.bias, proj1.weight")  # fmt: skip
        assert caplog.record_tuples[1] == ("fairseq2", logging.INFO, "Optimizer Parameter Group 1: proj2.bias, proj2.weight, proj3.bias, proj3.weight")  # fmt: skip

    def test_warns_when_group_has_no_match(self, caplog: Any) -> None:
        caplog.set_level(logging.INFO, logger="fairseq2")

        model = FooModel()

        configs = [
            FooParamGroupConfig(params=r"proj1\..*", lr=0.1),
            FooParamGroupConfig(params=r"proj4\..*", lr=0.2, weight_decay=0.3),
        ]

        parameters = list(prepare_parameter_groups(model, configs))

        assert len(parameters) == 3

        assert isinstance(parameters[0], dict)
        assert isinstance(parameters[1], dict)
        assert isinstance(parameters[2], dict)

        assert len(parameters[0]) == 2
        assert len(parameters[1]) == 3
        assert len(parameters[2]) == 1

        p = chain(
            model.proj2.parameters(),  # type: ignore[union-attr]
            model.proj3.parameters(),  # type: ignore[union-attr]
        )

        assert parameters[0]["params"] == list(model.proj1.parameters())  # type: ignore[union-attr]
        assert parameters[1]["params"] == list()
        assert parameters[2]["params"] == list(p)

        assert parameters[0]["lr"] == 0.1
        assert parameters[1]["lr"] == 0.2

        assert "lr" not in parameters[2]

        assert parameters[1]["weight_decay"] == 0.3

        assert "weight_decay" not in parameters[0]
        assert "weight_decay" not in parameters[2]

        assert "betas" not in parameters[0]
        assert "betas" not in parameters[1]
        assert "betas" not in parameters[2]

        assert len(caplog.records) == 3

        assert caplog.record_tuples[0] == ("fairseq2", logging.INFO, "Optimizer Parameter Group 0: proj1.bias, proj1.weight")  # fmt: skip
        assert caplog.record_tuples[1] == ("fairseq2", logging.WARN, "Optimizer parameter group 1 is empty.")  # fmt: skip
        assert caplog.record_tuples[2] == ("fairseq2", logging.INFO, "Optimizer Parameter Group 2: proj2.bias, proj2.weight, proj3.bias, proj3.weight")  # fmt: skip
