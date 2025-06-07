# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from typing import final

from typing_extensions import override


class ConfigMerger(ABC):
    @abstractmethod
    def merge(self, unstructure_config: object, overrides: object) -> object: ...


class ConfigMergeError(Exception):
    pass


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

                raise ConfigMergeError(
                    f"{pathname} at `overrides` must be of type `{list}`, but is of type `{type(del_keys)}` instead."
                )

            for idx, del_key in enumerate(del_keys):
                if not isinstance(del_key, str):
                    pathname = build_pathname("_del_")

                    raise ConfigMergeError(
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

                raise ConfigMergeError(
                    f"{pathname} at `overrides` must be of type `{Mapping}`, but is of type `{type(set_keys)}` instead."
                )

            for idx, (set_key, value) in enumerate(set_keys.items()):
                if not isinstance(set_key, str):
                    pathname = build_pathname("_set_")

                    raise ConfigMergeError(
                        f"Each key under {pathname} at `overrides` must be of type `{str}`, but the key at index {idx} is of type `{type(set_key)}` instead."
                    )

                output[set_key] = deepcopy(value)

        return output
