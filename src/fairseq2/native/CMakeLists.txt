# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ------------------------------------------------------------
# Target: fairseq2
# ------------------------------------------------------------

fairseq2_add_target(fairseq2 SHARED_LIBRARY)

target_sources(fairseq2
    PRIVATE
        data/data_source.cc
        data/fs.cc
        data/list_data_source.cc
    PUBLIC
        FILE_SET
            HEADERS
        FILES
            api.h
            data/data_source.h
            data/fs.h
            data/list_data_source.h
            error.h
        FILE_SET
            HEADERS
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

target_compile_features(fairseq2 PUBLIC cxx_std_17)

target_link_libraries(fairseq2 PRIVATE fmt::fmt natsort PUBLIC torch)

if(FAIRSEQ2_USE_CUDA)
    target_compile_features(fairseq2 PRIVATE cuda_std_17)

    target_compile_definitions(fairseq2 PRIVATE FAIRSEQ2_USE_CUDA)

    set(SUPPORTS_CUDA true)

    set(CUDA_VERSION_MAJOR ${CUDAToolkit_VERSION_MAJOR})
    set(CUDA_VERSION_MINOR ${CUDAToolkit_VERSION_MINOR})
else()
    set(SUPPORTS_CUDA false)

    set(CUDA_VERSION_MAJOR std::nullopt)
    set(CUDA_VERSION_MINOR std::nullopt)
endif()

configure_file(config.h.in config.h @ONLY)

add_library(fairseq2::fairseq2 ALIAS fairseq2)
