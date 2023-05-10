# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ------------------------------------------------------------
# Target: fairseq2
# ------------------------------------------------------------

add_library(fairseq2 SHARED)

target_sources(fairseq2
    PRIVATE
        error.cc
        exception.cc
        memory.cc
        py.cc
        data/batched_data_source.cc
        data/batched_by_length_data_source.cc
        data/data_pipeline.cc
        data/data_processor.cc
        data/data_source.cc
        data/file.cc
        data/file_stream.cc
        data/immutable_string.cc
        data/list_data_source.cc
        data/mapped_data_source.cc
        data/memory_stream.cc
        data/prefetched_data_source.cc
        data/record_reader.cc
        data/round_robin_data_source.cc
        data/sharded_data_source.cc
        data/shuffled_data_source.cc
        data/stream.cc
        data/tape.cc
        data/yielded_data_source.cc
        data/zipfile_data_source.cc
        data/zipped_data_source.cc
        data/detail/file.cc
        data/detail/file_system.cc
        data/text/text.cc
        data/text/text_data_source.cc
        data/text/text_line_reader.cc
        data/text/utf8_stream.cc
        data/text/detail/utf.cc
        data/text/sentencepiece/sp_decoder.cc
        data/text/sentencepiece/sp_encoder.cc
        data/text/sentencepiece/sp_model.cc
        data/text/sentencepiece/sp_processor.cc
)

fairseq2_set_compile_options(fairseq2)

target_compile_features(fairseq2 PUBLIC cxx_std_17)

if(PROJECT_IS_TOP_LEVEL)
    set(system)
else()
    set(system SYSTEM)
endif()

target_include_directories(fairseq2 ${system}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
)

target_link_libraries(fairseq2
    PRIVATE
        ${CMAKE_DL_LIBS}
    PRIVATE
        Iconv::Iconv
        TBB::tbb
        Threads::Threads
        fmt::fmt
        natsort
        sentencepiece-static
        kuba-zip
    PUBLIC
        torch
)

fairseq2_set_link_options(fairseq2)

if(FAIRSEQ2_INSTALL_STANDALONE)
    set_target_properties(fairseq2 PROPERTIES INSTALL_RPATH ${rpath_origin}/.)
else()
    set_target_properties(fairseq2 PROPERTIES
        VERSION
            ${PROJECT_VERSION}
        SOVERSION
            ${PROJECT_VERSION_MAJOR}
    )
endif()

add_library(fairseq2::fairseq2 ALIAS fairseq2)

install(
    TARGETS
        fairseq2
    EXPORT
        ${PROJECT_NAME}-targets
    LIBRARY
        DESTINATION
            ${install_lib_dir}
        COMPONENT
            runtime
        NAMELINK_COMPONENT
            devel
    ARCHIVE
        DESTINATION
            ${install_lib_dir}
        COMPONENT
            devel
    INCLUDES DESTINATION
        ${install_include_dir}
)

install(
    DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION
        ${install_include_dir}
    COMPONENT
        devel
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "extension" EXCLUDE
)

# ------------------------------------------------------------
# Library Configuration
# ------------------------------------------------------------

if(FAIRSEQ2_USE_CUDA)
    target_compile_features(fairseq2 PRIVATE cuda_std_17)

    target_compile_definitions(fairseq2 PRIVATE FAIRSEQ2_USE_CUDA)

    set(CUDA_VERSION_MAJOR "${CUDAToolkit_VERSION_MAJOR}")
    set(CUDA_VERSION_MINOR "${CUDAToolkit_VERSION_MINOR}")
else()
    set(CUDA_VERSION_MAJOR "std::nullopt")
    set(CUDA_VERSION_MINOR "std::nullopt")
endif()

configure_file(config.h.in config.h @ONLY)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/config.h
    DESTINATION
        ${install_include_dir}/native
    COMPONENT
        devel
)
