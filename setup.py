# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional, cast

import numpy
import torch
from setuptools import Command, find_packages, setup
from setuptools.command.install import install as install_base
from setuptools.dist import Distribution as DistributionBase
from setuptools.errors import FileError  # type: ignore[attr-defined]


class Distribution(DistributionBase):
    # We have to explicitly mark the distribution as non-pure since we will
    # inject our pre-built extension modules into it.
    def has_ext_modules(self) -> bool:
        return True


class install(install_base):
    distribution: Distribution

    install_base.sub_commands.append(("install_cmake", lambda self: True))

    # Old versions of distutils incorrectly check `ext_modules` to determine
    # whether a distribution is non-pure. We fix it here.
    def finalize_options(self) -> None:
        # Read `install_lib` before it gets overriden by the base method.
        can_override_lib = self.install_lib is None

        install_base.finalize_options(self)

        # Point `install_lib` to the right location.
        if can_override_lib and self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


# We inject our pre-built extension modules and optionally other related
# artifacts into the distribution by installing them via CMake.
class install_cmake(Command):
    verbose: bool

    description = "install CMake artifacts"

    user_options = [
        ("cmake-build-dir=", "b", "build directory (where to install from)"),
        ("install-dir=", "d", "directory to install to"),
    ]

    def initialize_options(self) -> None:
        self.cmake_build_dir = "build"

        self.install_dir: Optional[str] = None

    def finalize_options(self) -> None:
        self.ensure_dirname("cmake_build_dir")

        # If not specified, copy the value of `install_dir` from `install`
        # command's `install_lib` option.
        self.set_undefined_options("install", ("install_lib", "install_dir"))

        # Indicates whether to inject all CMake artifacts (e.g. shared
        # libraries) into the distribution.
        self.is_standalone = self._should_install_standalone()

    def _should_install_standalone(self) -> bool:
        try:
            f = open(os.path.join(self.cmake_build_dir, "CMakeCache.txt"))
        except FileNotFoundError:
            raise FileError("CMakeCache.txt is not found. Run CMake first.")

        # Read `FAIRSEQ2_INSTALL_STANDALONE` from cache.
        with f:
            for line in f:
                if line.startswith("FAIRSEQ2_INSTALL_STANDALONE"):
                    _, value = line.strip().split("=", 1)

                    return value.upper() in ["1", "ON", "TRUE", "YES", "Y"]

        return False

    def run(self) -> None:
        if self.is_standalone:
            self._cmake_install()

        self._cmake_install(component="python_modules")

    def _cmake_install(self, component: Optional[str] = None) -> None:
        cmd = ["cmake", "--install", self.cmake_build_dir]

        if component:
            cmd += ["--component", component]

        cmd += ["--prefix", cast(str, self.install_dir), "--strip"]

        if self.verbose:
            cmd += ["--verbose"]

        self.spawn(cmd)

    def get_outputs(self) -> List[str]:
        outputs = []

        install_cmd = self.get_finalized_command("install")

        # We have to strip the file paths to the installation root directory.
        if os.path.isabs(install_cmd.root):  # type: ignore[attr-defined]
            strip_idx = 0
        else:
            strip_idx = len(os.getcwd()) + 1

        # We extract the list of files from the CMake install manifests.
        def load_manifest(file: str) -> None:
            with open(os.path.join(self.cmake_build_dir, file)) as fp:
                for line in fp:
                    outputs.append(line[strip_idx:].rstrip())

        if self.is_standalone:
            load_manifest("install_manifest.txt")

        load_manifest("install_manifest_python_modules.txt")

        return outputs

    def get_inputs(self) -> List[str]:
        # We take no input from other commands.
        return []


setup(
    distclass=Distribution,
    cmdclass={
        "install": install,  # type: ignore[dict-item]
        "install_cmake": install_cmake,
    },
    name="fairseq2",
    version="0.1.0.dev0",
    description="FAIR Sequence Modeling Toolkit",
    url="https://github.com/facebookresearch/fairseq2",
    license="MIT",
    author="Fundamental AI Research (FAIR) at Meta",
    keywords=["machine learning"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"": ["py.typed", "*.pyi"]},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "overrides==7.3.1",
        # PyTorch has no ABI compatibility between releases; this means we have
        # to ensure that we depend on the exact same version that was used to
        # build our extension modules.
        "torch==" + torch.__version__,
        "typing_extensions>=4.3.0",
        # Runtime dependencies
        "func_argparse",
        "omegaconf",
        "overrides",
        "numpy==" + numpy.__version__,
        "sentencepiece",
        # TODO upgrade to 0.3.0 once they release it
        "torchtnt @ git+https://github.com/pytorch/tnt.git@1b71aecf3a2fb8204bf6010d3306d5ad9812bafd",
        "torchsnapshot",
        "torcheval",
        "wandb",
    ],
)
