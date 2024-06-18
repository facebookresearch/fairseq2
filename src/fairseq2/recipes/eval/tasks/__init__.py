# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

# flake8: noqa
from functools import partial
from itertools import product
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Tuple,
    Type,
    Union,
)

from ..api import Task, TaskConfig

if TYPE_CHECKING:
    from fairseq2.gang import Gang


class TaskRegistry:
    _REGISTRY: Dict[str, Callable[..., TaskConfig]] = {}

    @staticmethod
    def names() -> AbstractSet[str]:
        return TaskRegistry._REGISTRY.keys()

    @staticmethod
    def register(
        name: str,
        callable: Callable[..., TaskConfig],
    ) -> None:
        if name in TaskRegistry._REGISTRY:
            raise ValueError(f"Config for task {name} already exists.")
        TaskRegistry._REGISTRY[name] = callable

    @staticmethod
    def get_config(name: str, **kwargs: Any) -> TaskConfig:
        if name not in TaskRegistry._REGISTRY:
            raise ValueError(f"No task registered under the name {name}")
        return TaskRegistry._REGISTRY[name][0](**kwargs)

    @staticmethod
    def reset() -> None:
        TaskRegistry._REGISTRY = {}


def register_task(
    name: str,
    parameters: Optional[Dict[Union[str, Tuple[str, ...]], Iterable[Any]]] = None,
) -> Callable[[Callable[..., TaskConfig]], Callable[..., TaskConfig]]:
    """Register the task name with the decorated task configuration callable."""

    def register(callable: Callable[..., TaskConfig]) -> Callable[..., TaskConfig]:
        if parameters is None:
            TaskRegistry.register(name, callable)
        else:
            for values in product(*parameters.values()):
                param_dict: Dict[str, Any] = {}
                for keys, value in zip(parameters.keys(), values):
                    if isinstance(keys, tuple):
                        param_dict.update(zip(keys, value))
                    else:
                        param_dict[keys] = value
                task_name = name.format(**param_dict)
                TaskRegistry.register(task_name, partial(callable, **param_dict))
        return callable

    return register


def build_task(config: TaskConfig) -> Task:
    config_cls_name = config.__class__.__name__
    try:
        module = __import__(config.__class__.__module__, fromlist=[config_cls_name])
        cls_name = config.__class__.__name__.replace("Config", "")
        return getattr(module, cls_name).from_config(config)
    except ImportError:
        raise ValueError("No task class found for {config_cls_name}")


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("fairseq2.recipes.eval.tasks." + module)
