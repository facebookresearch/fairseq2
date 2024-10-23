#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

git clone --depth 1 --recurse-submodules --shallow-submodules --branch llvmorg-15.0.3\
    https://github.com/llvm/llvm-project.git /llvm

cmake\
    -GNinja\
    -S /llvm/llvm\
    -B /llvm-build\
    -DCMAKE_BUILD_TYPE=Release\
    -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;openmp"\
    -DLLVM_TARGETS_TO_BUILD=host\
    -Wno-dev

cmake --build /llvm-build && cmake --install /llvm-build

cp /llvm/clang/tools/clang-format/git-clang-format /usr/local/bin

rm -rf /llvm /llvm-build
