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
from typing import final

from torch import Tensor
from torch.nn import Parameter

from fairseq2.logging import log
from fairseq2.recipe.model import RecipeModel
from fairseq2.utils.validation import ValidationError


def prepare_parameter_groups(
    model: RecipeModel, groups: Sequence[ParameterGroup]
) -> Iterable[Tensor] | Iterable[dict[str, object]]:
    """
    A helper function for recipe optimizer factories that prepares the parameter
    groups to pass to the optimizer.

    :param model: The model that will be passed to the optimizer.
    :param groups: The list of groups from which to extract the parameters and
        other ``kwargs`` to pass to the optimizer.

    :returns: An :class:`Iterable` that can be passed as an argument to the
        ``params`` parameter of a PyTorch :class:`Optimizer`.

    .. note::

        Note that the order of groups is important when determining which
        parameter belongs to which group. Each parameter is assigned to the
        first group in the list that matches its name; therefore, it is
        essential to list the groups in the correct order.

    .. code-block:: python
        :caption: An example use of ``prepare_parameter_groups``

        from collections.abc import Sequence
        from dataclasses import dataclass, field

        from torch.optim import Optimizer

        from fairseq2.recipe import RecipeModel, TrainRecipe
        from fairseq2.recipe.component import register_component
        from fairseq2.recipe.config import Default
        from fairseq2.recipe.optim import ParameterGroup, prepare_parameter_groups
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
        class MyOptimizerGroupConfig:
            \"\"\"The parameter group configuration of MyOptimizer.\"\"\"

            params: str | Sequence[str] = ".*"
            \"\"\"The regular expression(s) to select the parameters belonging to this group.\"\"\"

            lr: float | Default = "default"
            \"\"\"If specified, overrides the top-level value.\"\"\"

            betas: tuple[float, float] | Default = "default"
            \"\"\"If specified, overrides the top-level value.\"\"\"


        class MyOptimizer(Optimizer):
            ...


        def create_my_optimizer(
            resolver: DependencyResolver, config: MyOptimizerConfig
        ) -> MyOptimizer:
            model = resolver.resolve(RecipeModel)

            # The list of configuration fields that parameter groups can override.
            fields = ["lr", "betas"]

            groups = [ParameterGroup(g.params, g, fields) for g in config.groups]

            # Converts groups to a form that can be passed to the optimizer.
            parameters = prepare_parameter_groups(model, groups)

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
    # If we don't have any parameter group descriptors, take the shortcut and
    # return the entire parameter list of the model as a single group.
    if not groups:
        return model.module.parameters()

    groups = list(groups)

    # Represents the fall-back group that holds the parameters whose name do not
    # match any pattern in `groups`.
    group = ParameterGroup([".*"], {}, [])

    groups.append(group)

    for param_name, param in model.module.named_parameters():
        for group in groups:
            if group.is_match(param_name):
                group.add_parameter(param_name, param)

                break

    output: list[dict[str, object]] = []

    for idx, group in enumerate(groups):
        if not group.empty:
            if group.is_fallback:
                continue

            log.warning("Optimizer parameter group {} is empty.", idx)
        elif log.is_enabled_for_info():
            s = ", ".join(sorted(n for n in group.param_names))

            log.info("Optimizer Parameter Group {}: {}", idx, s)

        kwargs = group.get_kwargs()

        output.append(kwargs)

    return output


@final
class ParameterGroup:
    """
    Represents an optimizer parameter group, used as input to the
    :func:`prepare_parameter_groups` function.
    """

    def __init__(
        self, name_patterns: str | Sequence[str], config: object, fields: Sequence[str]
    ) -> None:
        """
        :param name_patterns: The regular expression(s) used to select the
            parameters that belong to this group.
        :param config: An opaque object -*typically a dataclass*- that holds the
            configuration of this group.
        :param fields: The names of the configuration fields that ``config``
            holds. Any field that has a non-default value will be passed as a
            group ``kwarg`` to the optimizer.
        """
        if isinstance(name_patterns, str):
            name_patterns = [name_patterns]

        self._name_patterns = name_patterns
        self._config = config
        self._fields = fields
        self._params: list[Parameter] = []
        self._param_names: list[str] = []

    def is_match(self, name: str) -> bool:
        """
        Returns ``True`` if ``name`` matches one of the name patterns of this
        group.

        :param name: The name to check.
        """
        return any(name == p or re.match(p, name) for p in self._name_patterns)

    def add_parameter(self, param_name: str, param: Parameter) -> None:
        """
        Adds ``param`` to this group.

        :param param_name: The name of the parameter, used for logging and error
            reporting purposes.
        :param param: The parameter tensor.
        """
        self._params.append(param)

        self._param_names.append(param_name)

    def get_kwargs(self) -> dict[str, object]:
        """Returns the group ``kwargs`` to be passed to the optimizer."""
        kwargs: dict[str, object] = {"params": self._params}

        for field in self._fields:
            value = getattr(self._config, field)
            if value != "default":
                kwargs[field] = value

        return kwargs

    @property
    def param_names(self) -> Iterable[str]:
        """Gets the names of the parameters belonging to this group."""
        return self._param_names

    @property
    def empty(self) -> bool:
        """Gets whether this group has no parameters."""
        return bool(self._params)

    @property
    def is_fallback(self) -> bool:
        """Gets whether this is a catch-all fallback group."""
        return len(self._name_patterns) == 1 and self._name_patterns[0] == ".*"


def maybe_raise_param_group_length_error(
    field: str, value: Sequence[object], num_param_groups: int
) -> None:
    """
    A helper function that raises :class:`~fairseq2.utils.validation.ValidationError`
    if the length of a learning rate scheduler configuration field (``len(value)``)
    does not match the number of optimizer parameter groups (``num_param_groups``).

    :param field: The name of the configuration field that holds ``value``.
    :param value: The value whose length to check.
    :param num_param_groups: The number of optimizer parameter groups.

    :raises ~fairseq2.utils.validation.ValidationError: ``len(value)`` does not
        match ``num_param_groups``.

    .. code-block:: python
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
