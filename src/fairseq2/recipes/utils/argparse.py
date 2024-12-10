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
import yaml
from yaml.parser import ParserError

from fairseq2.typing import DataType


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
                try:
                    child_node = node[key]
                except KeyError:
                    child_node = None

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

                try:
                    del_keys = parent_node["_del_"]
                except KeyError:
                    del_keys = None

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
                    parsed_value = yaml.safe_load(value.lstrip())
                except ParserError:
                    raise ArgumentError(
                        self, f"invalid key-value pair: {item} (value must be yaml)"
                    )

                path = path.rstrip()

                if path.startswith("add:"):
                    path = path[4:]

                    directive = "_add_"
                elif path.startswith("set:"):
                    path = path[4:]

                    directive = "_set_"
                else:
                    directive = "_set_"

                parent_node, key = get_parent_node(path)

                try:
                    directive_keys = parent_node[directive]
                except KeyError:
                    directive_keys = None

                if not isinstance(directive_keys, dict):
                    directive_keys = {}

                    parent_node[directive] = directive_keys

                directive_keys[key] = parsed_value

        setattr(namespace, self.dest, data)


def parse_dtype(value: str) -> DataType:
    if value.startswith("torch."):
        value = value[6:]

    dtype = getattr(torch, value, None)

    if not isinstance(dtype, DataType):
        raise ArgumentTypeError("must be a `torch.dtype` identifier")

    return dtype
