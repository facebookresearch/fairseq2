# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import IO, Protocol, TypeAlias

import yaml
from yaml import YAMLError


class YamlLoader(Protocol):
    def __call__(self, input_: Path | IO[str]) -> list[object]:
        ...


class YamlDumper(Protocol):
    def __call__(self, obj: object, output: Path | IO[str]) -> None:
        ...


YamlError: TypeAlias = YAMLError


def load_yaml(input_: Path | IO[str]) -> list[object]:
    if isinstance(input_, Path):
        with input_.open() as fp:
            return load_yaml(fp)

    itr = yaml.safe_load_all(input_)

    return list(itr)


def dump_yaml(obj: object, output: Path | IO[str]) -> None:
    if isinstance(output, Path):
        with output.open("w") as fp:
            dump_yaml(obj, fp)
    else:
        yaml.safe_dump(obj, output, sort_keys=False)


def read_yaml(s: str) -> object:
    return yaml.safe_load(s)
