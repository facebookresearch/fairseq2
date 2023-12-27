# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

macro(fairseq2n_add_libpng)
    if(NOT TARGET png_static)
        set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)

        set(PNG_SHARED OFF)
        set(PNG_STATIC ON)
        set(PNG_TESTS  OFF)

        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
            set(PNG_INTEL_SSE on)
        elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
            set(PNG_ARM_NEON on)
        endif()

        set(SKIP_INSTALL_ALL TRUE)

        add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/libpng EXCLUDE_FROM_ALL)

        set_target_properties(png_static PROPERTIES
            C_VISIBILITY_PRESET
                hidden
            POSITION_INDEPENDENT_CODE
                ON
        )

        target_include_directories(png_static SYSTEM
            PUBLIC
                ${PROJECT_BINARY_DIR}/third-party/libpng
                ${PROJECT_SOURCE_DIR}/third-party/libpng
        )

        unset(SKIP_INSTALL_ALL)
    endif()
endmacro()
