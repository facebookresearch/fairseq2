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
        exception.cc
        memory.cc
        data/bucket_by_length_data_source.cc
        data/bucket_data_source.cc
        data/byte_stream.cc
        data/collater.cc
        data/data.cc
        data/data_length_extractor.cc
        data/data_pipeline.cc
        data/data_source.cc
        data/element_mapper.cc
        data/element_selector.cc
        data/file.cc
        data/file_stream.cc
        data/filter_data_source.cc
        data/immutable_string.cc
        data/list_data_source.cc
        data/map_data_source.cc
        data/memory_mapper.cc
        data/memory_stream.cc
        data/prefetch_data_source.cc
        data/py.cc
        data/record_reader.cc
        data/round_robin_data_source.cc
        data/shard_data_source.cc
        data/shuffle_data_source.cc
        data/skip_data_source.cc
        data/take_data_source.cc
        data/tape.cc
        data/yield_from_data_source.cc
        data/zip_data_source.cc
        data/zip_file_data_source.cc
        data/audio/audio_decoder.cc
        data/audio/detail/sndfile.cc
        data/detail/file.cc
        data/detail/file_system.cc
        data/text/string_splitter.cc
        data/text/string_to_int_converter.cc
        data/text/string_to_tensor_converter.cc
        data/text/text_data_source.cc
        data/text/text_line_reader.cc
        data/text/text_reader.cc
        data/text/utf8_stream.cc
        data/text/detail/utf.cc
        data/text/dict_tokenizer/dict_encoder.cc
        data/text/dict_tokenizer/dict_decoder.cc
        data/text/dict_tokenizer/dict_model.cc
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
        SndFile::sndfile
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
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(rpath_origin @loader_path)
    else()
        set(rpath_origin \$ORIGIN)
    endif()

    set_target_properties(fairseq2 PROPERTIES INSTALL_RPATH ${rpath_origin}/.)

    if(FAIRSEQ2_BUILD_FOR_WHEEL_BUNDLE)
        set_property(
            TARGET
                fairseq2
            APPEND PROPERTY
                INSTALL_RPATH
                    ${rpath_origin}/../../../..
                    ${rpath_origin}/../../torch/lib
        )
    endif()
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
        ${install_inc_dir}
)

install(
    DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION
        ${install_inc_dir}
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
        ${install_inc_dir}/native
    COMPONENT
        devel
)
