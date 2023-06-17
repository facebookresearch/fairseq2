# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from ctypes import CDLL, RTLD_GLOBAL
from pathlib import Path
from typing import Optional, Tuple

__version__ = "0.1.0.dev0"

# We import `torch` to ensure that libtorch.so is loaded into the process before
# our extension module.
import torch  # noqa: F401

# Holds the handle to the TBB shared library.
_tbb: Optional[CDLL] = None


def _load_tbb() -> None:
    # TODO: Do not hard-code the so name.
    if sys.platform == "darwin":
        dso_name = "libtbb.12.dylib"
    else:
        dso_name = "libtbb.so.12"

    global _tbb

    # If the system already provides TBB, skip the rest. The dynamic linker will
    # resolve it later. Do not use ctypes' find_library here since it hangs when
    # run under ThreadSanitizer.
    try:
        _tbb = CDLL(dso_name, mode=RTLD_GLOBAL)
    except OSError:
        pass
    else:
        return

    # Otherwise, load it from the tbb PyPI package if installed.
    lib_path = Path(sys.executable).parent.parent.joinpath("lib", dso_name)

    try:
        _tbb = CDLL(str(lib_path), mode=RTLD_GLOBAL)
    except OSError:
        raise RuntimeError(
            "Intel oneTBB is not found! Check your fairseq2 installation!"
        )


# We load TBB using our own lookup logic since it might be located in the
# Python virtual environment.
_load_tbb()

from fairseq2 import C  # type: ignore[attr-defined]


def get_lib() -> Path:
    """Return the directory that contains libfairseq2."""
    return Path(__file__).parent.joinpath("lib")


def get_include() -> Path:
    """Return the directory that contains libfairseq2 header files."""
    return Path(__file__).parent.joinpath("include")


def get_cmake_prefix_path() -> Path:
    """Return the directory that contains libfairseq2 CMake package."""
    return Path(__file__).parent.joinpath("lib", "cmake")


def supports_cuda() -> bool:
    """Return ``True`` if libfairseq2 supports CUDA."""
    return C._supports_cuda()  # type: ignore[no-any-return]


def cuda_version() -> Optional[Tuple[int, int]]:
    """Return the version of CUDA that libfairseq2 supports.

    :returns:
        The major and minor version segments.
    """
    return C._cuda_version()  # type: ignore[no-any-return]


# If ``True``, indicates that we are run under Sphinx.
_DOC_MODE = False
