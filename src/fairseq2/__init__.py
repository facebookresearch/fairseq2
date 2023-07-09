# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ctypes.util import find_library
from pathlib import Path
from typing import Optional, Tuple

__version__ = "0.1.0.dev0"


# Make sure that we have libsndfile on the system before loading our extension
# module.
libsndfile = find_library("sndfile")
if libsndfile is None:
    raise OSError(
        "libsndfile cannot be found on your system. Use your system package manager to install it (e.g. `apt install libsndfile1`)."
    )


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
