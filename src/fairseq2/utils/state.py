# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, Protocol, TypeVar, final, runtime_checkable

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override

from fairseq2.nn.utils.module import load_state_dict


@runtime_checkable
class Stateful(Protocol):
    """Represents an object that follows the ``state_dict`` convention."""

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None: ...


StatefulT = TypeVar("StatefulT")


class StatefulObjectBag:
    """Holds a collection of stateful objects."""

    _non_stateful_attrs: set[str]
    _explicit_stateful_attrs: dict[str, StateHandler[Any] | None]

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
                        state = self._state_dict(obj)
                    else:
                        state = obj
                else:
                    state = state_handler.get_state(obj)
            elif isinstance(obj, Stateful) and not self._is_dunder(name):
                state = self._state_dict(obj)
            else:
                continue

            state_dict[name] = state

        return state_dict

    @staticmethod
    def _state_dict(obj: Stateful) -> dict[str, object]:
        if isinstance(obj, FSDP):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message=r".*Please use DTensor instead.*"
                )
                warnings.filterwarnings(
                    action="ignore", message=r".*`_get_pg_default_device` will be deprecated.*"  # fmt: skip
                )

                return obj.state_dict()

        return obj.state_dict()

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
                            self._load_state_dict(obj, state)
                        except (ValueError, TypeError) as ex:
                            raise state_error(name, obj) from ex
                    else:
                        setattr(self, name, state)
                else:
                    try:
                        state_handler.set_state(obj, state)
                    except (ValueError, TypeError) as ex:
                        raise state_error(name, obj) from ex
            elif isinstance(obj, Stateful) and not self._is_dunder(name):
                try:
                    state = state_dict_.pop(name)
                except KeyError:
                    missing_stateful_attrs.append(name)

                    continue

                if not isinstance(state, Mapping):
                    raise state_type_error(name, state)

                try:
                    self._load_state_dict(obj, state)
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

    @staticmethod
    def _load_state_dict(obj: Stateful, state: Mapping[str, object]) -> None:
        if isinstance(obj, FSDP):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message=r".*Please use DTensor instead.*"
                )

                load_state_dict(obj, state)

            return

        if isinstance(obj, Module):
            load_state_dict(obj, state)

            return

        return obj.load_state_dict(state)

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
    def get_state(self, stateful: Optimizer) -> object:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message=r".*`_get_pg_default_device` will be deprecated.*"  # fmt: skip
            )
            warnings.filterwarnings(
                action="ignore", message=r".*You are using `torch\.load` with `weights_only=False`.*"  # fmt: skip
            )

            try:
                # FSDP uses warning level to dump a lot of noisy internal trace
                # information.
                logging.disable(logging.WARNING)

                return FSDP.optim_state_dict(self._module, stateful)
            except UnicodeDecodeError as ex:
                raise RuntimeError(
                    "FSDP has failed to gather the optimizer state with a pickling error. This might indicate a disk space issue. Make sure you have enough space on your file system. See the nested exception for details."
                ) from ex
            finally:
                logging.disable(logging.NOTSET)

    @override
    def set_state(self, stateful: Optimizer, state: object) -> None:
        if not isinstance(state, dict):
            raise TypeError(
                f"`state` must be of type `dict`, but is of type `{type(state)}` instead."
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message=r".*Please use DTensor instead.*"
            )

            try:
                # FSDP uses warning level to dump a lot of noisy internal trace
                # information.
                logging.disable(logging.WARNING)

                state_dict = FSDP.optim_state_dict_to_load(
                    self._module, stateful, state
                )
            finally:
                logging.disable(logging.NOTSET)

            stateful.load_state_dict(state_dict)
