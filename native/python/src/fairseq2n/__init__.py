# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import platform
import site
import sys
from ctypes import CDLL, RTLD_GLOBAL
from ctypes.util import find_library
from os import environ
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from fairseq2n.config import _CUDA_VERSION, _SUPPORTS_CUDA, _SUPPORTS_IMAGE

__version__ = "0.2.1.dev0"

# Indicates whether we are run under Sphinx.
DOC_MODE = False


# Keeps the shared libraries that we load using our own extended lookup logic
# in memory.
_libs: List[CDLL] = []


def _load_shared_libraries() -> None:
    # We import `torch` to ensure that libtorch and libtorch_python are loaded
    # into the process before our extension module.
    import torch

    # Intel oneTBB is only available on x86_64 systems.
    if platform.machine() == "x86_64":
        _load_tbb()

    _load_sndfile()


def _load_tbb() -> None:
    if platform.system() == "Darwin":
        lib_name = "libtbb.12.dylib"
    else:
        lib_name = "libtbb.so.12"

    libtbb = _load_shared_library(lib_name)
    if libtbb is None:
        raise OSError("Intel oneTBB is not found! Check your fairseq2 installation!")

    _libs.append(libtbb)


def _load_sndfile() -> None:
    if platform.system() == "Darwin":
        lib_name = "libsndfile.1.dylib"
    else:
        lib_name = "libsndfile.so.1"

    libsndfile = _load_shared_library(lib_name)
    if libsndfile is None:
        if "CONDA_PREFIX" in environ:
            raise OSError(
                "libsndfile is not found! Since you are in a Conda environment, use `conda install -c conda-forge libsndfile==1.0.31` to install it."
            )
        else:
            raise OSError(
                "libsndfile is not found! Use your system package manager to install it (e.g. `apt install libsndfile1`)."
            )

    _libs.append(libsndfile)


def _load_shared_library(lib_name: str) -> Optional[CDLL]:
    # In Conda environments, we always expect native libraries to be part of the
    # environment, so we skip the default lookup rules of the dynamic linker.
    if not "CONDA_PREFIX" in environ:
        try:
            # Use the global namespace to ensure that all modules use the same
            # library instance.
            return CDLL(lib_name, mode=RTLD_GLOBAL)
        except OSError:
            pass

        # On macOS, we also explicitly check the standard Homebrew locations.
        if platform.system() == "Darwin":
            for brew_path in ["/usr/local/lib", "/opt/homebrew/lib"]:
                try:
                    return CDLL(str(Path(brew_path, lib_name)), mode=RTLD_GLOBAL)
                except OSError:
                    pass

    if site.ENABLE_USER_SITE:
        site_packages = [site.getusersitepackages()]
    else:
        site_packages = []

    site_packages += site.getsitepackages()

    # If the system does not have the library, try to load it from the site
    # packages of the current Python environment.
    for packages_dir in site_packages:
        lib_path = Path(packages_dir).parent.parent.joinpath(lib_name)

        try:
            return CDLL(str(lib_path), mode=RTLD_GLOBAL)
        except OSError:
            pass

    return None


# We load shared libraries that we depend on using our own extended lookup logic
# since they might be located in non-default locations.
_load_shared_libraries()


def _check_cuda_runtime() -> None:
    if not _SUPPORTS_CUDA:
        return

    assert _CUDA_VERSION is not None

    major_cuda_ver, minor_cuda_ver = _CUDA_VERSION

    libcudart = _load_shared_library("libcudart.so")
    if libcudart is None:
        cuda = f"CUDA {major_cuda_ver}.{minor_cuda_ver}"

        raise OSError(
            f"fairseq2 is built with {cuda}, but {cuda} runtime cannot be found on your system. Either install {cuda} Toolkit or a CPU-only version of fairseq2 (see https://github.com/facebookresearch/fairseq2#variants)."
        )


_check_cuda_runtime()


def get_lib() -> Path:
    """Return the directory that contains fairseq2n shared library."""
    return Path(__file__).parent.joinpath("lib")


def get_include() -> Path:
    """Return the directory that contains fairseq2n header files."""
    return Path(__file__).parent.joinpath("include")


def get_cmake_prefix_path() -> Path:
    """Return the directory that contains fairseq2n CMake package."""
    return Path(__file__).parent.joinpath("lib/cmake")


def supports_image() -> bool:
    """Return ``True`` if fairseq2n supports JPEG/PNG decoding."""
    return _SUPPORTS_IMAGE


def supports_cuda() -> bool:
    """Return ``True`` if fairseq2n supports CUDA."""
    return _SUPPORTS_CUDA


def cuda_version() -> Optional[Tuple[int, int]]:
    """Return the version of CUDA that fairseq2n supports.

    :returns:
        The major and minor version segments.
    """
    return _CUDA_VERSION
