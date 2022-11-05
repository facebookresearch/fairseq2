# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import PurePath


def get_lib_path() -> PurePath:
    """Returns the path of the lib directory."""
    return PurePath(__file__).parent.parent.joinpath("lib")


def get_cmake_prefix_path() -> PurePath:
    """Returns the path of the CMake package directory."""
    return PurePath(__file__).parent.parent.joinpath("lib", "cmake")


__all__ = ["get_lib_path", "get_cmake_prefix_path"]
