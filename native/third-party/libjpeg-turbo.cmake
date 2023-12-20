# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

macro(fairseq2n_add_libjpeg_turbo)
    if(NOT TARGET jpeg_turbo_static)
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" AND NOT DEFINED ENV{ASM_NASM})
            find_program(NASM_EXECUTABLE NAMES nasm yasm)
            if(NOT NASM_EXECUTABLE)
                message(WARNING
                    "NASM or YASM compiler cannot be found. libjpeg-turbo won't have SIMD extensions enabled.")
            endif()
        endif()

        include(ExternalProject)

        set(prefix ${PROJECT_BINARY_DIR}/third-party/libjpeg-turbo)

        set(JPEG_TURBO_LIBRARY ${prefix}/lib/libturbojpeg.a)

        set(JPEG_TURBO_INCLUDE_DIR ${prefix}/include)

        ExternalProject_Add(
            #NAME
                jpeg_turbo
            PREFIX
                ${prefix}
            GIT_REPOSITORY
                https://github.com/libjpeg-turbo/libjpeg-turbo.git
            GIT_TAG
                3.0.1
            UPDATE_DISCONNECTED
                TRUE
            CMAKE_GENERATOR
                Ninja
            CMAKE_ARGS
                -DENABLE_SHARED=OFF
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                -DCMAKE_C_VISIBILITY_PRESET=hidden
                -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                -DCMAKE_INSTALL_LIBDIR=<INSTALL_DIR>/lib
                -DCMAKE_POLICY_DEFAULT_CMP0063=NEW
            BUILD_BYPRODUCTS
                ${JPEG_TURBO_LIBRARY}
        )

        file(MAKE_DIRECTORY ${JPEG_TURBO_INCLUDE_DIR})

        add_library(jpeg_turbo_static STATIC IMPORTED)

        add_dependencies(jpeg_turbo_static jpeg_turbo)

        set_property(TARGET jpeg_turbo_static PROPERTY IMPORTED_LOCATION ${JPEG_TURBO_LIBRARY})

        target_include_directories(jpeg_turbo_static INTERFACE ${JPEG_TURBO_INCLUDE_DIR})

        unset(prefix)
    endif()
endmacro()
