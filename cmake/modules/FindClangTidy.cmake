# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(FindPackageHandleStandardArgs)

find_program(CLANG_TIDY_EXECUTABLE NAMES clang-tidy)

mark_as_advanced(CLANG_TIDY_EXECUTABLE)

find_package_handle_standard_args(ClangTidy REQUIRED_VARS CLANG_TIDY_EXECUTABLE)
