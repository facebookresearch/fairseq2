# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

macro(fairseq2n_add_sentencepiece)
    if(NOT TARGET sentencepiece-static)
        set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        # Do not build the shared library.
        set(SPM_ENABLE_SHARED OFF)

        add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/sentencepiece EXCLUDE_FROM_ALL)

        target_compile_features(sentencepiece-static PRIVATE cxx_std_17)

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            # See https://github.com/protocolbuffers/protobuf/issues/6419.
            target_compile_options(sentencepiece-static PRIVATE -Wno-stringop-overflow)
        endif()

        if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            target_compile_options(sentencepiece-static PRIVATE -Wno-deprecated-declarations)
        endif()

        set_target_properties(sentencepiece-static PROPERTIES
            CXX_VISIBILITY_PRESET
                hidden
            POSITION_INDEPENDENT_CODE
                ON
        )

        target_include_directories(sentencepiece-static SYSTEM
            PUBLIC
                ${PROJECT_SOURCE_DIR}/third-party
                ${PROJECT_SOURCE_DIR}/third-party/sentencepiece/third_party/protobuf-lite
        )

        # We depend on the phony torch_cxx11_abi target to ensure that we use the
        # same libstdc++ ABI as PyTorch.
        target_link_libraries(sentencepiece-static PRIVATE torch_cxx11_abi)
    endif()
endmacro()
