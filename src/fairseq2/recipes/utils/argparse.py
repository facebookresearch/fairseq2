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
    """Adds support for reading key-value pairs in format ``<key>=<yaml_value>``."""

    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        help: str | None = None,
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

        for item in values:
            key_value = item.split("=", maxsplit=1)
            if len(key_value) != 2:
                raise ArgumentError(self, f"invalid key-value pair: {item}")

            key, value = [kv.strip() for kv in key_value]

            try:
                parsed_value = yaml.safe_load(value)
            except ParserError:
                raise ArgumentError(
                    self, f"invalid key-value pair: {item} (value must be yaml)"
                )

            fields = key.split(".")

            tmp = data

            for field in fields[:-1]:
                try:
                    d = tmp[field]
                except KeyError:
                    d = None

                if not isinstance(d, dict):
                    d = {}

                    tmp[field] = d

                tmp = d

            tmp[fields[-1]] = parsed_value

        setattr(namespace, self.dest, data)


def parse_dtype(value: str) -> DataType:
    """Parse ``value`` as a  ``torch.dtype``."""
    if value.startswith("torch."):
        value = value[6:]

    if isinstance(dtype := getattr(torch, value, None), DataType):
        return dtype

    raise ArgumentTypeError("must be a `torch.dtype` identifier")
