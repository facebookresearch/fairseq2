# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from argparse import (
    ZERO_OR_MORE,
    Action,
    ArgumentError,
    ArgumentParser,
    ArgumentTypeError,
    Namespace,
)
from typing import Any, final

import torch
from ruamel.yaml import YAML

from fairseq2.data_type import DataType
from fairseq2.utils.yaml import YamlError


@final
class ConfigAction(Action):
    """
    Adds support for reading configuration key-value pairs in format ``<key>=<yaml_value>``.
    """

    def __init__(
        self, option_strings: list[str], dest: str, help: str | None = None
    ) -> None:
        super().__init__(
            option_strings,
            nargs=ZERO_OR_MORE,
            help=help,
            metavar="NAME=VALUE",
            dest=dest,
        )

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        data: dict[str, Any] = {}

        def get_parent_node(path: str) -> tuple[dict[str, Any], str]:
            keys = path.split(".")

            node = data

            for key in keys[:-1]:
                child_node = node.get(key)

                if not isinstance(child_node, dict):
                    child_node = {}

                    node[key] = child_node

                node = child_node

            return node, keys[-1]

        for item in values:
            item = item.strip()

            if item.startswith("del:"):
                path = item[4:]

                if "=" in path:
                    raise ArgumentError(self, f"key should not contain '=': {item}")

                parent_node, key = get_parent_node(path)

                del_keys = parent_node.get("_del_")

                if not isinstance(del_keys, list):
                    del_keys = []

                    parent_node["_del_"] = del_keys

                del_keys.append(key)
            else:
                path_value = item.split("=", maxsplit=1)
                if len(path_value) != 2:
                    raise ArgumentError(self, f"invalid key-value pair: {item}")

                path, value = path_value

                try:
                    parsed_value = self._read_yaml(value.lstrip())
                except YamlError:
                    raise ArgumentError(
                        self, f"invalid key-value pair: {item} (value must be yaml)"
                    )

                path = path.rstrip()

                if path.startswith("set:"):
                    path = path[4:]

                parent_node, key = get_parent_node(path)

                set_keys = parent_node.get("_set_")

                if not isinstance(set_keys, dict):
                    set_keys = {}

                    parent_node["_set_"] = set_keys

                set_keys[key] = parsed_value

        items = getattr(namespace, self.dest, None)
        if items is None:
            items = []

        items.append(data)

        setattr(namespace, self.dest, items)

    @staticmethod
    def _read_yaml(s: str) -> object:
        yaml = YAML(typ="safe", pure=True)

        return yaml.load(s)


def parse_dtype(value: str) -> DataType:
    if value.startswith("torch."):
        value = value[6:]

    dtype = getattr(torch, value, None)

    if not isinstance(dtype, DataType):
        raise ArgumentTypeError("must be a torch.dtype identifier")

    return dtype
