# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import platform
import site
from ctypes import CDLL, RTLD_GLOBAL
from os import environ
from pathlib import Path
from typing import List, Optional, Tuple

from fairseq2n.config import (
    _CUDA_VERSION,
    _SUPPORTS_CUDA,
    _SUPPORTS_IMAGE,
    _TORCH_VARIANT,
    _TORCH_VERSION,
)

__version__ = "0.3.0.dev0"


def get_lib() -> Path:
    """Return the directory that contains fairseq2n shared library."""
    return Path(__file__).parent.joinpath("lib")


def get_include() -> Path:
    """Return the directory that contains fairseq2n header files."""
    return Path(__file__).parent.joinpath("include")


def get_cmake_prefix_path() -> Path:
    """Return the directory that contains fairseq2n CMake package."""
    return Path(__file__).parent.joinpath("lib/cmake")


def torch_version() -> str:
    """Return the version of PyTorch that was used to build fairseq2n."""
    return _TORCH_VERSION


def torch_variant() -> str:
    """Return the variant of PyTorch that was used to build fairseq2n."""
    return _TORCH_VARIANT


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
        raise OSError(
            "fairseq2 requires Intel oneTBB which is normally installed along with fairseq2 as a dependency. Check your environment and reinstall fairseq2 if necessary."
        )

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
                "fairseq2 requires libsndfile. Since you are in a Conda environment, use `conda install -c conda-forge libsndfile==1.0.31` to install it."
            )
        else:
            raise OSError(
                "fairseq2 requires libsndfile. Use your system package manager to install it (e.g. `apt install libsndfile1`)."
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

        # On macOS, we also explicitly check the well-known Homebrew locations.
        if platform.system() == "Darwin":
            for brew_path in ["/usr/local/lib", "/opt/homebrew/lib", "~/homebrew"]:
                path = Path(brew_path, lib_name).expanduser()

                try:
                    return CDLL(str(path), mode=RTLD_GLOBAL)
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


def _check_torch_version() -> None:
    import torch

    # Trim the local version label.
    source_version = torch.__version__.split("+", 1)[0]

    if source_var := torch.version.cuda:
        # Use only the major and minor version segments.
        source_variant = "CUDA " + ".".join(source_var.split(".", 2)[:2])
    else:
        source_variant = "CPU-only"

    if source_version != _TORCH_VERSION or source_variant != _TORCH_VARIANT:
        raise RuntimeError(
            f"fairseq2 requires a {_TORCH_VARIANT} build of PyTorch {_TORCH_VERSION}, but the installed version is a {source_variant} build of PyTorch {source_version}. Either follow the instructions at https://pytorch.org/get-started/locally to update PyTorch, or the instructions at https://github.com/facebookresearch/fairseq2#variants to update fairseq2."
        )


_check_torch_version()
