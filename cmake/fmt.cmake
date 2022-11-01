# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include_guard(DIRECTORY)

function(fairseq2_find_fmt version)
    # Treat bundled fmt as a system dependency.
    set(FMT_SYSTEM_HEADERS ON)

    fairseq2_find_package(fmt ${version})

    # If we are not using the bundled fmt, skip the rest.
    if(NOT FMT_SOURCE_DIR)
        return()
    endif()

    target_compile_features(fmt PRIVATE cxx_std_17)

    set_target_properties(fmt PROPERTIES
        CXX_VISIBILITY_PRESET
            hidden
        POSITION_INDEPENDENT_CODE
            ON
    )

    # We depend on the phony torch_cxx11_abi target to ensure that we use the
    # same libstdc++ ABI as PyTorch.
    target_link_libraries(fmt torch_cxx11_abi)
endfunction()
