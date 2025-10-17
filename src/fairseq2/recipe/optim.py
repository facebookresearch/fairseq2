# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides helper functions to support the addition of new optimizers
and learning rate schedulers in recipes.

Functions
^^^^^^^^^

* :func:`prepare_parameter_groups`
* :func:`maybe_raise_param_group_length_error`
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields

from torch import Tensor
from torch.nn import Parameter

from fairseq2.logging import log
from fairseq2.recipe.config import ParameterGroupConfig, default
from fairseq2.recipe.model import RecipeModel
from fairseq2.utils.validation import ValidationError


def prepare_parameter_groups(
    model: RecipeModel, group_configs: Sequence[ParameterGroupConfig]
) -> Iterable[Tensor] | Iterable[dict[str, object]]:
    """
    Prepares the parameter groups to pass to an optimizer based on the specified
    model and group recipe configurations.

    Returns an :class:`Iterable` that can be passed as an argument to the
    ``params`` parameter of a PyTorch :class:`Optimizer`.

    Fields in `group_configs` whose value is set to :data:`default` will use the
    default configuration in the corresponding top-level configuration. For
    instance, if :attr:`AdamWGroupConfig.betas` is set to :data:`default`, the
    optimizer will use the value of :attr:`AdamWConfig.betas`.

    Note that the order of groups is important when determining which parameter
    belongs to which group. Each parameter is assigned to the first group in the
    list that matches its name; therefore, it is essential to list the groups in
    the correct order.

    .. code:: python
        :caption: An example use of ``prepare_parameter_groups``

        from collections.abc import Sequence
        from dataclasses import dataclass, field

        from torch.optim import Optimizer

        from fairseq2.recipe import RecipeModel, TrainRecipe
        from fairseq2.recipe.component import register_component
        from fairseq2.recipe.config import Default, ParameterGroupConfig, default
        from fairseq2.recipe.optim import prepare_parameter_groups
        from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver

        @dataclass
        class MyOptimizerConfig:
            \"\"\"The top-level recipe configuration of MyOptimizer.\"\"\"

            lr: float = 0.1
            \"\"\"The default top-level learning rate.\"\"\"

            betas: tuple[float, float] = (0.9, 0.99)
            \"\"\"The default top-level beta values.\"\"\"

            groups: Sequence[MyOptimizerGroupConfig] = field(default_factory=list)
            \"\"\"The configuration of individual parameter groups.\"\"\"


        @dataclass
        class MyOptimizerGroupConfig(ParameterGroupConfig):
            \"\"\"The parameter group configuration of MyOptimizer.\"\"\"

            lr: float | Default = default
            \"\"\"If specified, overrides the top-level value.\"\"\"

            betas: tuple[float, float] | Default = default
            \"\"\"If specified, overrides the top-level value.\"\"\"


        class MyOptimizer(Optimizer):
            ...


        def create_my_optimizer(
            resolver: DependencyResolver, config: MyOptimizerConfig
        ) -> MyOptimizer:
            model = resolver.resolve(RecipeModel)

            # Converts group configurations to an iterable of parameter groups
            # that can be passed to an optimizer.
            parameters = prepare_parameter_groups(model, config.groups)

            # Initialize the optimizer with `parameters`.
            return MyOptimizer(parameters, config.lr, config.betas)


        class MyTrainRecipe(TrainRecipe):
            def register(self, container: DependencyContainer) -> None:
                register_component(
                    container,
                    Optimizer,
                    name="my_optimizer",
                    config_kls=MyOptimizerConfig,
                    factory=create_my_optimizer,
                )

            ...
    """
    # If we don't have any parameter group configurations, take the shortcut and
    # return the entire parameter list of the model as a single group.
    if not group_configs:
        return model.module.parameters()

    groups = []

    for config in group_configs:
        name_patterns = config.params
        if isinstance(name_patterns, str):
            name_patterns = [name_patterns]

        kwargs: dict[str, object] = {}

        for field in fields(config):
            if field.name == "params":
                continue

            value = getattr(config, field.name)
            if value == default:
                continue

            kwargs[field.name] = value

        group = _ParameterGroup(name_patterns, kwargs, [], [])

        groups.append(group)

    # Represents the fall-back group that holds the parameters whose names do
    # not match any group patterns.
    group = _ParameterGroup([".*"], {}, [], [])

    groups.append(group)

    for name, param in model.module.named_parameters():
        for group in groups:
            if any(name == p or re.match(p, name) for p in group.name_patterns):
                group.params.append(param)

                group.param_names.append(name)

                break

    output: list[dict[str, object]] = []

    for idx, group in enumerate(groups):
        if not group.params:
            # If `True`, means fall-back group.
            if len(group.name_patterns) == 1 and group.name_patterns[0] == ".*":
                continue

            log.warning("Optimizer parameter group {} is empty.", idx)
        elif log.is_enabled_for_info():
            s = ", ".join(sorted(n for n in group.param_names))

            log.info("Optimizer Parameter Group {}: {}", idx, s)

        output.append(group.kwargs)

    return output


# Used by `prepare_parameter_groups` for internal bookkeeping.
@dataclass
class _ParameterGroup:
    name_patterns: Sequence[str]
    kwargs: dict[str, object]
    params: list[Parameter]
    param_names: list[str]

    def __post_init__(self) -> None:
        self.kwargs["params"] = self.params


def maybe_raise_param_group_length_error(
    field: str, value: Sequence[object], num_param_groups: int
) -> None:
    """
    Raises :class:`~fairseq2.utils.validation.ValidationError` if the length of
    a learning rate scheduler configuration field (i.e. ``len(value)``) does not
    match the number of optimizer parameter groups.

    :raises ~fairseq2.utils.validation.ValidationError: If ``len(value)`` does
        not match ``num_param_groups``.

    .. code:: python
        :caption: A basic use of ``maybe_raise_param_group_length_error``

        from torch.optim import Optimizer

        from fairseq2.recipe.config import MyleLRConfig
        from fairseq2.recipe.optim import maybe_raise_param_group_length_error

        def get_start_lr(config: MyleLRConfig, optimizer: Optimizer) -> list[float]:
            num_param_groups = len(optimizer.param_groups)

            start_lr: float | list[float] = config.start_lr

            if isinstance(start_lr, float):
                return [start_lr] * num_param_groups

            maybe_raise_param_group_length_error("start_lr", start_lr, num_param_groups)

            return start_lr
    """
    if len(value) != num_param_groups:
        raise ValidationError(
            f"The length of `{field}` must match the number of optimizer parameter groups ({num_param_groups}), but is {len(value)} instead.", field="lr_scheduler.config"  # fmt: skip
        )
