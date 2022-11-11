# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include_guard(DIRECTORY)

include(GoogleTest)

function(fairseq2_find_gtest)
    set(INSTALL_GTEST OFF)

    fairseq2_find_package(gtest)

    # If we are not using the bundled gtest, skip the rest.
    if(NOT gtest_SOURCE_DIR)
        return()
    endif()

    # We depend on the phony torch_cxx11_abi target to ensure that we use the
    # same libstdc++ ABI as PyTorch.
    target_link_libraries(gtest      PRIVATE torch_cxx11_abi)
    target_link_libraries(gtest_main PRIVATE torch_cxx11_abi)
endfunction()
