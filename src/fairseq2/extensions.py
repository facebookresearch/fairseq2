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


def run_extensions(
    extension_name: str, signature: str, *args: Any, **kwargs: Any
) -> None:
    should_trace = "FAIRSEQ2_EXTENSION_TRACE" in os.environ

    for entry_point in entry_points(group=extension_name):
        try:
            extension = entry_point.load()

            extension(*args, **kwargs)
        except TypeError as ex:
            # Not ideal, but there is no other way to find out whether the error
            # was raised due to a wrong extension signature.
            if not str(ex).startswith(f"{entry_point.attr}()"):
                raise

            if should_trace:
                raise ExtensionError(
                    entry_point.value, f"The '{entry_point.value}' entry point cannot be run as an extension function since its signature does not match `{signature}`."  # fmt: skip
                ) from None

            log.warning("'{}' entry point is not a valid extension function. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value)  # fmt: skip
        except Exception as ex:
            if should_trace:
                raise ExtensionError(
                    entry_point.value, f"'{entry_point.value}' extension function has failed. See the nested exception for details."  # fmt: skip
                ) from ex

            log.warning("'{}' extension function failed. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value)  # fmt: skip

        if should_trace:
            log.info("`{}` extension function run successfully.", entry_point.value)  # fmt: skip


class ExtensionError(Exception):
    entry_point: str

    def __init__(self, entry_point: str, message: str) -> None:
        super().__init__(message)

        self.entry_point = entry_point
