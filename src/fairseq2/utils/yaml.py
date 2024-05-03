# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from pathlib import PosixPath, PurePath
from typing import Any, Callable, Type, TypeVar

from yaml.nodes import Node, ScalarNode
from yaml.representer import SafeRepresenter

from fairseq2.typing import DataType

T = TypeVar("T")


def represent_as_str(representer: SafeRepresenter, data: Any) -> ScalarNode:
    return representer.represent_str(str(data))


def represent_as_enum(representer: SafeRepresenter, data: Enum) -> ScalarNode:
    return representer.represent_str(data.name)


def register_yaml_representer(
    kls: Type[T], representer: Callable[[SafeRepresenter, T], Node]
) -> None:
    if kls not in SafeRepresenter.yaml_representers:
        SafeRepresenter.add_representer(kls, representer)


def _register_yaml_representers() -> None:
    for kls in [DataType, PurePath, PosixPath]:
        register_yaml_representer(kls, represent_as_str)
