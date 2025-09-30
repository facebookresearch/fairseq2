# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os

from importlib_metadata import entry_points

from fairseq2.logging import log
from fairseq2.runtime.dependency import DependencyContainer


def _register_extensions(container: DependencyContainer) -> None:
    should_trace = "FAIRSEQ2_EXTENSION_TRACE" in os.environ

    for entry_point in entry_points(group="fairseq2.extension"):
        try:
            extension = entry_point.load()

            extension(container)
        except TypeError as ex:
            # Not ideal, but there is no other way to find out whether the error
            # was raised due to a wrong function signature.
            if not str(ex).startswith(f"{entry_point.attr}()"):
                raise

            if should_trace:
                msg = f"{entry_point.value} entry point cannot be run as an extension since its signature does not match `extension_function(container: DependencyContainer)`."

                raise ExtensionError(entry_point.value, msg) from None

            log.warning("{} entry point is not a valid extension. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value)  # fmt: skip
        except Exception as ex:
            if should_trace:
                msg = f"{entry_point.value} extension failed."

                raise ExtensionError(entry_point.value, msg) from ex

            log.warning("{} extension failed. Set `FAIRSEQ2_EXTENSION_TRACE` environment variable to print the stack trace.", entry_point.value)  # fmt: skip

        if should_trace:
            log.info("{} extension registered successfully.", entry_point.value)  # fmt: skip


class ExtensionError(Exception):
    def __init__(self, entry_point: str, message: str) -> None:
        super().__init__(message)

        self.entry_point = entry_point
