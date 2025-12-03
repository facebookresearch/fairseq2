# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import final, overload

from typing_extensions import override

from fairseq2.error import FormatError
from fairseq2.runtime.dependency import get_dependency_resolver


def get_env() -> Environment:
    resolver = get_dependency_resolver()

    return resolver.resolve(Environment)


class Environment(ABC, Iterable[tuple[str, str]]):
    @abstractmethod
    def get(self, name: str) -> str: ...

    @overload
    def maybe_get(self, name: str) -> str | None: ...

    @overload
    def maybe_get(self, name: str, default: str) -> str: ...

    @abstractmethod
    def maybe_get(self, name: str, default: str | None = None) -> str | None: ...

    @abstractmethod
    def set(self, name: str, value: str) -> None: ...

    @abstractmethod
    def has(self, name: str) -> bool: ...

    @abstractmethod
    def to_dict(self) -> dict[str, str]: ...

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[str, str]]: ...


class EnvironmentVariableError(Exception):
    def __init__(self, var_name: str, message: str) -> None:
        super().__init__(message)

        self.var_name = var_name


@final
class StandardEnvironment(Environment):
    @override
    def get(self, name: str) -> str:
        return os.environ[name]

    @overload
    def maybe_get(self, name: str) -> str | None: ...

    @overload
    def maybe_get(self, name: str, default: str) -> str: ...

    @override
    def maybe_get(self, name: str, default: str | None = None) -> str | None:
        return os.environ.get(name, default)

    @override
    def set(self, name: str, value: str) -> None:
        os.environ[name] = value

    @override
    def has(self, name: str) -> bool:
        return name in os.environ

    @override
    def to_dict(self) -> dict[str, str]:
        return dict(os.environ)

    @override
    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(os.environ.items())


def get_rank(env: Environment) -> int:
    """
    :raises LookupError:
    :raises FormatError:
    """
    rank = maybe_get_rank(env)
    if rank is None:
        raise LookupError()

    return rank


def maybe_get_rank(env: Environment) -> int | None:
    """
    :raises FormatError:
    """
    return _maybe_get_rank(env, "RANK")


def get_world_size(env: Environment) -> int:
    """
    :raises LookupError:
    :raises FormatError:
    """
    world_size = maybe_get_world_size(env)
    if world_size is None:
        raise LookupError()

    return world_size


def maybe_get_world_size(env: Environment) -> int | None:
    """
    :raises FormatError:
    """
    return _maybe_get_world_size(env, "WORLD_SIZE")


def get_local_rank(env: Environment) -> int:
    """
    :raises LookupError:
    :raises FormatError:
    """
    rank = maybe_get_local_rank(env)
    if rank is None:
        raise LookupError()

    return rank


def maybe_get_local_rank(env: Environment) -> int | None:
    """
    :raises FormatError:
    """
    return _maybe_get_rank(env, "LOCAL_RANK")


def get_local_world_size(env: Environment) -> int:
    """
    :raises LookupError:
    :raises FormatError:
    """
    world_size = maybe_get_local_world_size(env)
    if world_size is None:
        raise LookupError()

    return world_size


def maybe_get_local_world_size(env: Environment) -> int | None:
    """
    :raises FormatError:
    """
    return _maybe_get_world_size(env, "LOCAL_WORLD_SIZE")


def _maybe_get_rank(env: Environment, var_name: str) -> int | None:
    s = env.maybe_get(var_name)
    if s is None:
        return None

    try:
        value = int(s)
    except ValueError:
        raise FormatError(f"Expected to be an integer, but is '{s} instead.") from None

    if value < 0:
        raise FormatError(
            f"Expected to be equal or greater than 0, but is {value} instead."
        )

    return value


def _maybe_get_world_size(env: Environment, var_name: str) -> int | None:
    s = env.maybe_get(var_name)
    if s is None:
        return None

    try:
        value = int(s)
    except ValueError:
        raise FormatError(f"Expected to be an integer, but is '{s} instead.") from None

    if value < 1:
        raise FormatError(
            f"Expected to be equal or greater than 1, but is {value} instead."
        )

    return value
