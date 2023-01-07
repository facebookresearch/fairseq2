# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from ctypes import CDLL, RTLD_GLOBAL
from ctypes.util import find_library
from pathlib import PurePath
from typing import Optional

# Holds the handle to the TBB shared library.
_tbb: Optional[CDLL] = None


def _load() -> None:
    filename = find_library("tbb")

    if sys.platform == "darwin":
        dso_name = "libtbb.12.dylib"
    else:
        dso_name = "libtbb.so.12"

    # If the system already provides TBB, skip the rest. The dynamic linker will
    # resolve it later.
    if filename == dso_name:
        return

    global _tbb

    # Otherwise, load it from the tbb PyPI package if installed.
    lib_path = PurePath(sys.executable).parent.parent.joinpath("lib", dso_name)

    try:
        _tbb = CDLL(str(lib_path), mode=RTLD_GLOBAL)
    except OSError:
        raise RuntimeError(
            "Intel oneTBB is not found! Check your fairseq2 installation!"
        )
