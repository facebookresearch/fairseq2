# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import final

from typing_extensions import override


class ConfigMerger(ABC):
    @abstractmethod
    def merge(self, unstructure_config: object, overrides: object) -> object: ...


@final
class StandardConfigMerger(ConfigMerger):
    @override
    def merge(self, unstructured_config: object, overrides: object) -> object:
        if not isinstance(overrides, Mapping):
            return overrides

        if not isinstance(unstructured_config, Mapping):
            unstructured_config = {}

        return self._merge_map(unstructured_config, overrides, [])

    @classmethod
    def _merge_map(
        cls, target: Mapping[str, object], source: Mapping[str, object], path: list[str]
    ) -> Mapping[str, object]:
        def build_pathname(subpath: str) -> str:
            return ".".join(path + [subpath])

        output = {}

        ignored_keys = set()

        del_keys = source.get("_del_")
        if del_keys is not None:
            if not isinstance(del_keys, list):
                pathname = build_pathname("_del_")

                raise TypeError(
                    f"{pathname} at `overrides` must be of type `{list}`, but is of type `{type(del_keys)}` instead."
                )

            for idx, del_key in enumerate(del_keys):
                if not isinstance(del_key, str):
                    pathname = build_pathname("_del_")

                    raise TypeError(
                        f"Each element under {pathname} at `overrides` must be of type `{str}`, but the element at index {idx} is of type `{type(del_key)}` instead."
                    )

                ignored_keys.add(del_key)

        for key, target_value in target.items():
            if key not in ignored_keys:
                output[key] = deepcopy(target_value)

        for key, source_value in source.items():
            if key == "_del_" or key == "_set_":
                continue

            if not isinstance(source_value, Mapping):
                output[key] = deepcopy(source_value)

                continue

            target_value = output.get(key)
            if not isinstance(target_value, Mapping):
                target_value = {}

            path.append(key)

            output[key] = cls._merge_map(target_value, source_value, path)

            path.pop()

        set_keys = source.get("_set_")
        if set_keys is not None:
            if not isinstance(set_keys, Mapping):
                pathname = build_pathname("_set_")

                raise TypeError(
                    f"{pathname} at `overrides` must be of type `{Mapping}`, but is of type `{type(set_keys)}` instead."
                )

            for idx, (set_key, value) in enumerate(set_keys.items()):
                if not isinstance(set_key, str):
                    pathname = build_pathname("_set_")

                    raise TypeError(
                        f"Each key under {pathname} at `overrides` must be of type `{str}`, but the key at index {idx} is of type `{type(set_key)}` instead."
                    )

                output[set_key] = deepcopy(value)

        return output


class ConfigProcessor(ABC):
    @abstractmethod
    def process(
        self,
        unstructured_config: object,
        directives: Sequence[ConfigDirective] | None = None,
    ) -> object: ...


@final
class StandardConfigProcessor(ConfigProcessor):
    def __init__(self, default_directives: Sequence[ConfigDirective]) -> None:
        self._default_directives = default_directives

    @override
    def process(
        self,
        unstructured_config: object,
        directives: Sequence[ConfigDirective] | None = None,
    ) -> object:
        if not self._default_directives and not directives:
            return unstructured_config

        return self._do_process(
            unstructured_config, unstructured_config, directives or []
        )

    def _do_process(
        self,
        value: object,
        unstructured_config: object,
        directives: Sequence[ConfigDirective],
    ) -> object:
        if isinstance(value, list):
            return [self._do_process(e, unstructured_config, directives) for e in value]

        if isinstance(value, dict):
            output = {}

            for k, v in value.items():
                output[k] = self._do_process(v, unstructured_config, directives)

            return output

        if isinstance(value, str):
            for directive in chain(self._default_directives, directives):
                obj = directive.execute(value, unstructured_config)
                if not isinstance(obj, str):
                    return obj

                value = obj

            return value

        return value


class ConfigDirective(ABC):
    @abstractmethod
    def execute(self, value: str, unstructured_config: object) -> object: ...


class ConfigDirectiveError(Exception):
    pass


@final
class ReplacePathDirective(ConfigDirective):
    def __init__(self, config_path: Path) -> None:
        self._config_path = str(config_path)

    @override
    def execute(self, value: str, unstructured_config: object) -> object:
        return value.replace("${dir}", self._config_path)
