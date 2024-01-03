# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    final,
    runtime_checkable,
)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Module
from torch.optim import Optimizer

from fairseq2.typing import finaloverride


@runtime_checkable
class Stateful(Protocol):
    """Represents an object that follows the ``state_dict`` convention."""

    def state_dict(self) -> Dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        ...


class StatefulObjectBag:
    """Holds a collection of stateful objects."""

    stateful_objects: Dict[str, Tuple[Any, Optional[StateHandler]]]

    def __init__(self) -> None:
        super().__setattr__("stateful_objects", {})

    def __getattr__(self, name: str) -> Any:
        if "stateful_objects" in self.__dict__ and name in self.stateful_objects:
            return self.stateful_objects[name][0]

        raise AttributeError(
            f"`{type(self).__name__}` object has no attribute '{name}'."
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.stateful_objects:
            _, state_handler = self.stateful_objects[name]

            self.stateful_objects[name] = (value, state_handler)
        elif name not in self.__dict__ and isinstance(value, Stateful):
            self.register_stateful(name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self.stateful_objects:
            del self.stateful_objects[name]
        else:
            super().__delattr__(name)

    def register_stateful(
        self, name: str, obj: Any, state_handler: Optional[StateHandler] = None
    ) -> None:
        """Add ``obj`` to the bag and preserve its state in ``state_dict``.

        :param name:
            The attribute name to refer to ``obj``.
        :param obj:
            The object to add.
        :param state_handler:
            The handler to load and extract the state of ``obj``. If ``None``
            and ``obj`` is of type :class:`Stateful`, then its ``state_dict``
            will be used; otherwise, ``obj`` will be preserved as is.
        """
        if hasattr(self, name):
            raise AttributeError(
                f"`{type(self).__name__}` object already has an attribute '{name}'."
            )

        self.stateful_objects[name] = (obj, state_handler)

    def register_non_stateful(self, name: str, obj: Any) -> None:
        """Add ``obj`` to the bag, but do not preserve its state in ``state_dict``.

        :param name:
            The attribute name to refer to ``obj``.
        :param obj:
            The object to add.
        """
        if hasattr(self, name):
            raise AttributeError(
                f"`{type(self).__name__}` object already has an attribute '{name}'."
            )

        super().__setattr__(name, obj)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {}

        state: Any

        for name, (stateful, state_handler) in self.stateful_objects.items():
            if state_handler is not None:
                state = state_handler.extract_state(stateful)
            elif isinstance(stateful, Stateful):
                state = stateful.state_dict()
            else:
                state = stateful

            state_dict[name] = state

        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if self.stateful_objects.keys() != state_dict.keys():
            raise ValueError(
                f"`state_dict` must contain items {list(self.stateful_objects.keys())}, but contains {list(state_dict.keys())} instead."
            )

        for name, (stateful, state_handler) in self.stateful_objects.items():
            state = state_dict[name]

            if state_handler is not None:
                state_handler.load_state(stateful, state)
            elif isinstance(stateful, Stateful):
                stateful.load_state_dict(state)
            else:
                self.stateful_objects[name] = (state, None)


class StateHandler(ABC):
    """Loads and extracts the state of an object registered with an instance of
    :class:`StatefulObjectBag`."""

    @abstractmethod
    def load_state(self, stateful: Any, state: Any) -> None:
        """Load the state of ``stateful`` from ``state``."""

    @abstractmethod
    def extract_state(self, stateful: Any) -> Any:
        """Extract the state of ``stateful``."""


@final
class FSDPOptimizerStateHandler(StateHandler):
    """Loads and extracts the state of an :class:`Optimizer` managed by FSDP."""

    module: Module

    def __init__(self, module: Module) -> None:
        """
        :param module:
            The module that is of type :class:`FSDP` or contains a module that
            is of type :class:`FSDP`.
        """
        self.module = module

    @finaloverride
    def load_state(self, stateful: Optimizer, state: Any) -> None:
        state_dict = FSDP.optim_state_dict_to_load(self.module, stateful, state)

        stateful.load_state_dict(state_dict)

    @finaloverride
    def extract_state(self, stateful: Optimizer) -> Any:
        return FSDP.optim_state_dict(self.module, stateful)
