# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    Mapping,
    Optional,
    Protocol,
    Set,
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

    _non_stateful_attrs: Set[str]
    _explicit_stateful_attrs: Dict[str, Optional[StateHandler[Any]]]

    def __init__(self) -> None:
        super().__init__()  # play nicely as a mixin.

        self._non_stateful_attrs = set()

        self._explicit_stateful_attrs = {}

    def __delattr__(self, name: str) -> None:
        try:
            self._non_stateful_attrs.remove(name)
        except KeyError:
            pass

        try:
            del self._explicit_stateful_attrs[name]
        except KeyError:
            pass

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
        try:
            self._non_stateful_attrs.remove(name)
        except KeyError:
            pass

        self._explicit_stateful_attrs[name] = state_handler

        setattr(self, name, obj)

    @final
    def register_non_stateful(self, name: str, obj: Any) -> None:
        """Add ``obj`` to the bag, but do not preserve its state in ``state_dict``.

        :param name:
            The attribute name to refer to ``obj``.
        :param obj:
            The object to add.
        """
        try:
            del self._explicit_stateful_attrs[name]
        except KeyError:
            pass

        self._non_stateful_attrs.add(name)

        setattr(self, name, obj)

    @final
    def state_dict(self) -> Dict[str, Any]:
        state_dict = {}

        state: Any

        for name, obj in self.__dict__.items():
            if name in self._non_stateful_attrs:
                continue

            is_explicit, state_handler = self._is_explicit(name)

            if is_explicit:
                if state_handler is None:
                    if isinstance(obj, Stateful):
                        state = obj.state_dict()
                    else:
                        state = obj
                else:
                    state = state_handler.get_state(obj)
            elif isinstance(obj, Stateful) and not self._is_dunder(name):
                state = obj.state_dict()
            else:
                continue

            state_dict[name] = state

        return state_dict

    @final
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        missing_stateful_attrs = []

        state_dict_ = dict(state_dict)

        for name, obj in self.__dict__.items():
            if name in self._non_stateful_attrs:
                continue

            is_explicit, state_handler = self._is_explicit(name)

            if is_explicit:
                try:
                    state = state_dict_.pop(name)
                except KeyError:
                    missing_stateful_attrs.append(name)

                    continue

                if state_handler is None:
                    if isinstance(obj, Stateful):
                        obj.load_state_dict(state)
                    else:
                        setattr(self, name, state)
                else:
                    state_handler.set_state(obj, state)
            elif isinstance(obj, Stateful) and not self._is_dunder(name):
                try:
                    state = state_dict_.pop(name)
                except KeyError:
                    missing_stateful_attrs.append(name)

                    continue

                obj.load_state_dict(state)

        if missing_stateful_attrs:
            missing_stateful_attrs.sort()

            raise ValueError(
                f"`state_dict` must contain the states of the following attributes: {', '.join(missing_stateful_attrs)}"
            )

        if state_dict_:
            extra_keys = list(state_dict_.keys())

            extra_keys.sort()

            raise ValueError(
                f"`state_dict` must only contain the states of the attributes of this object, but it contains the following extra keys: {', '.join(extra_keys)}"
            )

    def _is_explicit(self, name: str) -> Tuple[bool, Optional[StateHandler[Any]]]:
        try:
            state_handler = self._explicit_stateful_attrs[name]

            return True, state_handler
        except KeyError:
            pass

        return False, None

    @staticmethod
    def _is_dunder(name: str) -> bool:
        return len(name) > 4 and name.startswith("__") and name.endswith("__")


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
        try:
            # PyTorch 2.2 wrongfully uses warning level to dump a lot of noisy
            # internal trace information.
            logging.disable(logging.WARNING)

            return FSDP.optim_state_dict(self._module, stateful)
        finally:
            logging.disable(logging.NOTSET)

    @override
    def set_state(self, stateful: Optimizer, state: Any) -> None:
        try:
            # PyTorch 2.2 wrongfully uses warning level to dump a lot of noisy
            # internal trace information.
            logging.disable(logging.WARNING)

            state_dict = FSDP.optim_state_dict_to_load(self._module, stateful, state)
        finally:
            logging.disable(logging.NOTSET)

        stateful.load_state_dict(state_dict)
