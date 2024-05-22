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
    Generic,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    final,
    runtime_checkable,
)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Module
from torch.optim import Optimizer

from fairseq2.typing import override


@runtime_checkable
class Stateful(Protocol):
    """Represents an object that follows the ``state_dict`` convention."""

    def state_dict(self) -> Dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        ...


StatefulT = TypeVar("StatefulT")


class StatefulObjectBag:
    """Holds a collection of stateful objects."""

    _stateful_objects: Dict[str, Tuple[Any, Optional[StateHandler[Any]]]]

    def __init__(self) -> None:
        super().__setattr__("_stateful_objects", {})

    def __getattr__(self, name: str) -> Any:
        if "_stateful_objects" in self.__dict__ and name in self._stateful_objects:
            return self._stateful_objects[name][0]

        raise AttributeError(
            f"`{type(self).__name__}` object has no attribute '{name}'."
        )

    def __setattr__(self, name: str, value: Any) -> None:
        # TODO: fix!
        if name == "__orig_class__":
            super().__setattr__(name, value)
            return

        if name in self._stateful_objects:
            _, state_handler = self._stateful_objects[name]

            self._stateful_objects[name] = (value, state_handler)
        elif name not in self.__dict__ and isinstance(value, Stateful):
            self.register_stateful(name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._stateful_objects:
            del self._stateful_objects[name]
        else:
            super().__delattr__(name)

    @final
    def register_stateful(
        self,
        name: str,
        obj: StatefulT,
        state_handler: Optional[StateHandler[StatefulT]] = None,
    ) -> None:
        """Add ``obj`` to the bag and preserve its state in ``state_dict``.

        :param name:
            The attribute name to refer to ``obj``.
        :param obj:
            The object to add.
        :param state_handler:
            The handler to get and set the state of ``obj``. If ``None`` and
            ``obj`` is of type :class:`Stateful`, then its ``state_dict`` will
            be used; otherwise, ``obj`` will be preserved as is.
        """
        if hasattr(self, name):
            raise AttributeError(
                f"`{type(self).__name__}` object already has an attribute '{name}'."
            )

        self._stateful_objects[name] = (obj, state_handler)

    @final
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

    @final
    def state_dict(self) -> Dict[str, Any]:
        state_dict = {}

        state: Any

        for name, (stateful, state_handler) in self._stateful_objects.items():
            if state_handler is not None:
                state = state_handler.get_state(stateful)
            elif isinstance(stateful, Stateful):
                state = stateful.state_dict()
            else:
                state = stateful

            state_dict[name] = state

        return state_dict

    @final
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if self._stateful_objects.keys() != state_dict.keys():
            raise ValueError(
                f"`state_dict` must contain items {list(self._stateful_objects.keys())}, but contains {list(state_dict.keys())} instead."
            )

        for name, (stateful, state_handler) in self._stateful_objects.items():
            state = state_dict[name]

            if state_handler is not None:
                state_handler.set_state(stateful, state)
            elif isinstance(stateful, Stateful):
                stateful.load_state_dict(state)
            else:
                self._stateful_objects[name] = (state, None)


class StateHandler(ABC, Generic[StatefulT]):
    """Gets and sets the state of an object registered with an instance of
    :class:`StatefulObjectBag`."""

    @abstractmethod
    def get_state(self, stateful: StatefulT) -> Any:
        """Get the state of ``stateful``."""

    @abstractmethod
    def set_state(self, stateful: StatefulT, state: Any) -> None:
        """Set the state of ``stateful`` to ``state``."""


@final
class FSDPOptimizerStateHandler(StateHandler[Optimizer]):
    """Gets and sets the state of an :class:`Optimizer` managed by FSDP."""

    _module: Module

    def __init__(self, module: Module) -> None:
        """
        :param module:
            The module that is of type :class:`FSDP` or contains a module that
            is of type :class:`FSDP`.
        """
        self._module = module

    @override
    def get_state(self, stateful: Optimizer) -> Any:
        return FSDP.optim_state_dict(self._module, stateful)

    @override
    def set_state(self, stateful: Optimizer, state: Any) -> None:
        state_dict = FSDP.optim_state_dict_to_load(self._module, stateful, state)

        stateful.load_state_dict(state_dict)
