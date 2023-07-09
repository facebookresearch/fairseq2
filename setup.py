# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from os import path
from typing import Final, List, Optional

import torch
from packaging import version
from setuptools import Command, find_packages, setup
from setuptools.command.install import install as install_base
from setuptools.dist import Distribution as DistributionBase
from setuptools.errors import FileError  # type: ignore[attr-defined]


class Distribution(DistributionBase):
    # We have to explicitly mark the distribution as non-pure since we will
    # inject our pre-built extension module into it.
    def has_ext_modules(self) -> bool:
        return True


class install(install_base):
    distribution: Distribution

    install_base.sub_commands.append(("install_cmake", lambda self: True))

    # Old versions of distutils incorrectly check `ext_modules` to determine
    # whether a distribution is non-pure. We fix it here.
    def finalize_options(self) -> None:
        # Check `install_lib` before it gets overriden by the base.
        can_override_lib = self.install_lib is None

        install_base.finalize_options(self)

        # Set `install_lib` to the right location if allowed.
        if can_override_lib and self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


class install_cmake(Command):
    cmake_build_dir: str
    install_dir: str
    bundle_lib: bool
    verbose: bool

    description: Final = "install CMake artifacts"

    user_options: Final = [
        ("cmake-build-dir=", "b", "build directory (where to install from)"),
        ("install-dir=", "d", "directory to install to"),
    ]

    def initialize_options(self) -> None:
        self.cmake_build_dir = "build"

        self.install_dir = None  # type: ignore[assignment]

    def finalize_options(self) -> None:
        self.ensure_dirname("cmake_build_dir")

        self.set_undefined_options("install", ("install_lib", "install_dir"))

        # Check if we should bundle our shared library with the distribution.
        self.bundle_lib = self._should_bundle_lib()

    def _should_bundle_lib(self) -> bool:
        try:
            f = open(path.join(self.cmake_build_dir, "CMakeCache.txt"))
        except FileNotFoundError:
            raise FileError("CMakeCache.txt is not found. Run CMake first.")

        with f:
            for line in f:
                if line.startswith("FAIRSEQ2_BUILD_FOR_WHEEL_BUNDLE"):
                    _, value = line.strip().split("=", 1)

                    return value.upper() in ["1", "ON", "TRUE", "YES", "Y"]

        return False

    def run(self) -> None:
        if self.bundle_lib:
            self._cmake_install()

        self._cmake_install(component="python")

    def _cmake_install(self, component: Optional[str] = None) -> None:
        cmd = ["cmake", "--install", self.cmake_build_dir]

        if component:
            cmd += ["--component", component]

        cmd += ["--prefix", self.install_dir, "--strip"]

        if self.verbose:
            cmd += ["--verbose"]

        self.spawn(cmd)

    def get_outputs(self) -> List[str]:
        outputs = []

        if self.bundle_lib:
            manifests = ["install_manifest.txt"]
        else:
            manifests = []

        manifests.append("install_manifest_python.txt")

        # We have to strip the file paths to the install directory to be
        # consistent with the setuptools commands.
        strip_length = len(path.abspath(self.install_dir)) + 1

        # Extract the list of files from the CMake install manifests.
        for m in manifests:
            with open(path.join(self.cmake_build_dir, m)) as fp:
                for pathname in fp:
                    pathname = pathname[strip_length:].rstrip()

                    outputs.append(path.join(self.install_dir, pathname))

        return outputs

    def get_inputs(self) -> List[str]:
        # We take no input.
        return []


dependencies = [
    "jiwer~=3.0",
    "numpy~=1.23",
    "overrides~=7.3",
    "packaging~=23.1",
    "pyyaml~=6.0",
    "sacrebleu~=2.3",
    "tbb==2021.9.0",
    # PyTorch has no ABI compatibility between releases; this means we have to
    # ensure that we depend on the exact same version that was used to build our
    # extension module.
    "torch==" + version.parse(torch.__version__).public,
    "torcheval~=0.0.6",
    "typing_extensions~=4.3",
]

cli_dependencies = [
    "func_argparse~=1.1",
    "omegaconf~=2.2",
    "submitit~=1.4",
    "torchsnapshot~=0.1.0",
    "torchtnt~=0.0.7",
    "torchx~=0.5",
    "wandb~=0.13",
]

all_dependencies = dependencies + cli_dependencies


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
    package_data={"": ["py.typed", "*.pyi", "assets/cards/*.yaml"]},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=dependencies,
    extras_require={
        "cli": cli_dependencies,
        "all": all_dependencies,
    },
    entry_points={
        "console_scripts": [
            "fairseq2 = fairseq2.cli.commands:main",
        ],
    },
)
