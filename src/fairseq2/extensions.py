# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from typing import Any

from importlib_metadata import entry_points

from fairseq2.logging import log


def run_extensions(name: str, *args: Any, **kwargs: Any) -> None:
    should_trace = "FAIRSEQ2_EXTENSION_TRACE" in os.environ

    for entry_point in entry_points(group="fairseq2.extension"):
        try:
            extension_module = entry_point.load()
        except Exception as ex:
            if should_trace:
                raise ExtensionError(
                    f"The `{entry_point.value}` fairseq2 extension module has failed to load. See the nested exception for details."
                ) from ex

            log.warning("The `{}` fairseq2 extension module has failed to load. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value)  # fmt: skip

            continue

        try:
            extension = getattr(extension_module, name)
        except AttributeError:
            continue

        try:
            extension(*args, **kwargs)
        except Exception as ex:
            if should_trace:
                raise ExtensionError(
                    f"The `{entry_point.value}.{name}` fairseq2 extension function has failed. See the nested exception for details."
                ) from ex

            log.warning("The `{}.{}` fairseq2 extension function has failed. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value, name)  # fmt: skip

            continue

        if should_trace:
            log.info("The `{}.{}` fairseq2 extension function run successfully.", entry_point.value, name)  # fmt: skip


class ExtensionError(Exception):
    pass