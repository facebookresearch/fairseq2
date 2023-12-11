# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

macro(fairseq2n_add_gtest)
    if(NOT TARGET GTest::gtest_main)
        set(INSTALL_GTEST OFF)

        add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/gtest EXCLUDE_FROM_ALL)

        # We depend on the phony torch_cxx11_abi target to ensure that we use
        # the same libstdc++ ABI as PyTorch.
        target_link_libraries(gtest PRIVATE torch_cxx11_abi)
    endif()

    include(GoogleTest)
endmacro()
