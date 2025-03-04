# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, Protocol, TypeVar, final, runtime_checkable


@runtime_checkable
class Stateful(Protocol):
    """Represents an object that follows the ``state_dict`` convention."""

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None: ...


StatefulT = TypeVar("StatefulT")


class StatefulObjectBag:
    """Holds a collection of stateful objects."""

    _explicit_only: bool
    _non_stateful_attrs: set[str]
    _explicit_stateful_attrs: dict[str, StateHandler[Any] | None]

    def __init__(self, explicit_only: bool = False) -> None:
        super().__init__()  # play nicely as a mixin.

        self._explicit_only = explicit_only

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
        state_handler: StateHandler[StatefulT] | None = None,
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
    def register_non_stateful(self, name: str, obj: object) -> None:
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
    def state_dict(self) -> dict[str, object]:
        state_dict = {}

        state: object

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
            elif self._explicit_only:
                continue
            elif isinstance(obj, Stateful) and not self._is_dunder(name):
                state = obj.state_dict()
            else:
                continue

            state_dict[name] = state

        return state_dict

    @final
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        def state_error(name: str, obj: object) -> ValueError:
            return ValueError(
                f"`state_dict['{name}']` is not a valid `{type(obj)}` state. See the nested exception for details."
            )

        def state_type_error(name: str, state: object) -> TypeError:
            return TypeError(
                f"`state_dict['{name}']` must be of type `{Mapping}`, but is of type `{type(state)}` instead."
            )

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
                        if not isinstance(state, Mapping):
                            raise state_type_error(name, state)

                        try:
                            obj.load_state_dict(state)
                        except (ValueError, TypeError) as ex:
                            raise state_error(name, obj) from ex
                    else:
                        setattr(self, name, state)
                else:
                    try:
                        state_handler.set_state(obj, state)
                    except (ValueError, TypeError) as ex:
                        raise state_error(name, obj) from ex
            elif self._explicit_only:
                continue
            elif isinstance(obj, Stateful) and not self._is_dunder(name):
                try:
                    state = state_dict_.pop(name)
                except KeyError:
                    missing_stateful_attrs.append(name)

                    continue

                if not isinstance(state, Mapping):
                    raise state_type_error(name, state)

                try:
                    obj.load_state_dict(state)
                except (ValueError, TypeError) as ex:
                    raise state_error(name, obj) from ex

        if missing_stateful_attrs:
            missing_stateful_attrs.sort()

            s = ", ".join(missing_stateful_attrs)

            raise ValueError(
                f"`state_dict` must contain the states of all of the following attribute(s): {s}"
            )

        if state_dict_:
            s = ", ".join(sorted(state_dict_.keys()))

            raise ValueError(
                f"`state_dict` must contain only the states of the attributes of this object, but it contains the following unexpected keys: {s}"
            )

    def _is_explicit(self, name: str) -> tuple[bool, StateHandler[Any] | None]:
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
    def get_state(self, stateful: StatefulT) -> object:
        """Get the state of ``stateful``."""

    @abstractmethod
    def set_state(self, stateful: StatefulT, state: object) -> None:
        """Set the state of ``stateful`` to ``state``."""
