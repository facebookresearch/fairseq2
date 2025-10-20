# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings


def enable_deprecation_warnings() -> None:
    """
    Enables fairseq2 deprecation warnings.

    This function is called by all recipe entry points to ensure that
    deprecation warnings are visible to users. It is strongly advised that
    developers also call it in their test runners to ensure that all deprecation
    warnings are caught and handled in a timely manner.
    """
    warnings.filterwarnings("once", category=DeprecationWarning, module="fairseq2.*")


def _warn_deprecated(msg: str) -> None:
    warnings.warn(msg, DeprecationWarning, stacklevel=2)


# TODO: Remove in v0.13
def _warn_progress_deprecated(value: bool | None) -> None:
    if value is not None:
        _warn_deprecated(
            "`progress` parameter in all fairseq2 APIs is deprecated, has no effect, and will be removed in v0.13."
        )
