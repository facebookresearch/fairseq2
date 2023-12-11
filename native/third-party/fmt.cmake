# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

macro(fairseq2n_add_fmt)
    if(NOT TARGET fmt::fmt)
        set(FMT_SYSTEM_HEADERS ON)

        add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/fmt EXCLUDE_FROM_ALL)

        target_compile_features(fmt PRIVATE cxx_std_17)

        set_target_properties(fmt PROPERTIES
            CXX_VISIBILITY_PRESET
                hidden
            POSITION_INDEPENDENT_CODE
                ON
        )

        # We depend on the phony torch_cxx11_abi target to ensure that we use
        # the same libstdc++ ABI as PyTorch.
        target_link_libraries(fmt PRIVATE torch_cxx11_abi)
    endif()
endmacro()
