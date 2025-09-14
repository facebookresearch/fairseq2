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

from fairseq2.recipe.config import Default
from fairseq2.recipe.optim import ParameterGroup, prepare_parameter_groups
from tests.unit.recipe.helpers import create_foo_model


@dataclass
class FooParamGroupConfig:
    lr: float | Default = "default"
    betas: tuple[float, float] | Default = "default"
    weight_decay: float | Default = "default"


class TestPrepareParameterGroups:
    def test_works(self, caplog: Any) -> None:
        caplog.set_level(logging.INFO, logger="fairseq2")

        model = create_foo_model()

        fields = ["lr", "betas", "weight_decay"]

        config1 = FooParamGroupConfig(lr=0.1)
        config2 = FooParamGroupConfig(lr=0.2, weight_decay=0.3)

        groups = [
            ParameterGroup(r"proj1\..*", config1, fields),
            ParameterGroup([r"proj2\.weight", r"proj2\.bias"], config2, fields),
        ]

        parameters = list(prepare_parameter_groups(model, groups))

        assert len(parameters) == 3

        assert isinstance(parameters[0], dict)
        assert isinstance(parameters[1], dict)
        assert isinstance(parameters[2], dict)

        assert len(parameters[0]) == 2
        assert len(parameters[1]) == 3
        assert len(parameters[2]) == 1

        assert parameters[0]["params"] == list(model.module.proj1.parameters())  # type: ignore[union-attr]
        assert parameters[1]["params"] == list(model.module.proj2.parameters())  # type: ignore[union-attr]
        assert parameters[2]["params"] == list(model.module.proj3.parameters())  # type: ignore[union-attr]

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

        model = create_foo_model()

        parameters = prepare_parameter_groups(model, groups=[])

        assert list(parameters) == list(model.module.parameters())

        assert len(caplog.records) == 0

    def test_works_when_groups_are_exhaustive(self, caplog: Any) -> None:
        caplog.set_level(logging.INFO, logger="fairseq2")

        model = create_foo_model()

        fields = ["lr", "betas", "weight_decay"]

        config1 = FooParamGroupConfig(lr=0.1)
        config2 = FooParamGroupConfig(lr=0.2, weight_decay=0.3)

        groups = [
            ParameterGroup(r"proj1\..*", config1, fields),
            ParameterGroup(r"proj.*", config2, fields),
        ]

        parameters = list(prepare_parameter_groups(model, groups))

        assert len(parameters) == 2

        assert isinstance(parameters[0], dict)
        assert isinstance(parameters[1], dict)

        assert len(parameters[0]) == 2
        assert len(parameters[1]) == 3

        p = chain(
            model.module.proj2.parameters(),  # type: ignore[union-attr]
            model.module.proj3.parameters(),  # type: ignore[union-attr]
        )

        assert parameters[0]["params"] == list(model.module.proj1.parameters())  # type: ignore[union-attr]
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

        model = create_foo_model()

        fields = ["lr", "betas", "weight_decay"]

        config1 = FooParamGroupConfig(lr=0.1)
        config2 = FooParamGroupConfig(lr=0.2, weight_decay=0.3)

        groups = [
            ParameterGroup(r"proj1\..*", config1, fields),
            ParameterGroup(r"proj4\..*", config2, fields),
        ]

        parameters = list(prepare_parameter_groups(model, groups))

        assert len(parameters) == 3

        assert isinstance(parameters[0], dict)
        assert isinstance(parameters[1], dict)
        assert isinstance(parameters[2], dict)

        assert len(parameters[0]) == 2
        assert len(parameters[1]) == 3
        assert len(parameters[2]) == 1

        p = chain(
            model.module.proj2.parameters(),  # type: ignore[union-attr]
            model.module.proj3.parameters(),  # type: ignore[union-attr]
        )

        assert parameters[0]["params"] == list(model.module.proj1.parameters())  # type: ignore[union-attr]
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
