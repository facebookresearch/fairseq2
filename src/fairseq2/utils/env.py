# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import final

from typing_extensions import override


class Environment(ABC):
    @abstractmethod
    def get(self, name: str) -> str: ...

    @abstractmethod
    def maybe_get(self, name: str) -> str | None: ...

    @abstractmethod
    def set(self, name: str, value: str) -> None: ...

    @abstractmethod
    def has(self, name: str) -> bool: ...


@final
class StandardEnvironment(Environment):
    @override
    def get(self, name: str) -> str:
        return os.environ[name]

    @override
    def maybe_get(self, name: str) -> str | None:
        return os.environ.get(name)

    @override
    def set(self, name: str, value: str) -> None:
        os.environ[name] = value

    @override
    def has(self, name: str) -> bool:
        return name in os.environ


class EnvironmentVariableError(Exception):
    def __init__(self, var_name: str, message: str) -> None:
        super().__init__(message)

        self.var_name = var_name


def get_rank(env: Environment) -> int | None:
    return _get_rank(env, "RANK")


def get_world_size(env: Environment) -> int | None:
    return _get_world_size(env, "WORLD_SIZE")


def get_local_rank(env: Environment) -> int | None:
    return _get_rank(env, "LOCAL_RANK")


def get_local_world_size(env: Environment) -> int | None:
    return _get_world_size(env, "LOCAL_WORLD_SIZE")


def _get_rank(env: Environment, var_name: str) -> int | None:
    s = env.maybe_get(var_name)
    if s is None:
        return None

    try:
        value = int(s)
    except ValueError:
        raise EnvironmentVariableError(
            var_name, f"{var_name} environment variable cannot be parsed as an integer."
        ) from None

    if value < 0:
        raise EnvironmentVariableError(
            var_name, f"{var_name} environment variable is expected to be greater than or equal to 0, but is {value} instead."  # fmt: skip
        )

    return value


def _get_world_size(env: Environment, var_name: str) -> int | None:
    s = env.maybe_get(var_name)
    if s is None:
        return None

    try:
        value = int(s)
    except ValueError:
        raise EnvironmentVariableError(
            var_name, f"{var_name} environment variable cannot be parsed as an integer."
        ) from None

    if value < 1:
        raise EnvironmentVariableError(
            var_name, f"{var_name} environment variable is expected to be greater than or equal to 1, but is {value} instead."  # fmt: skip
        )

    return value
