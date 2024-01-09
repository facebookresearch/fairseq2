# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ------------------------------------------------------------
# Target: fairseq2n
# ------------------------------------------------------------

add_library(fairseq2n SHARED)

target_sources(fairseq2n
    PRIVATE
        exception.cc
        memory.cc
        data/bucket_by_length_data_source.cc
        data/bucket_data_source.cc
        data/byte_stream.cc
        data/collater.cc
        data/concat_data_source.cc
        data/constant_data_source.cc
        data/count_data_source.cc
        data/data.cc
        data/data_length_extractor.cc
        data/data_pipeline.cc
        data/data_source.cc
        data/element_mapper.cc
        data/element_selector.cc
        data/file.cc
        data/file_mapper.cc
        data/file_stream.cc
        data/filter_data_source.cc
        data/immutable_string.cc
        data/list_data_source.cc
        data/map_data_source.cc
        data/memory_stream.cc
        data/prefetch_data_source.cc
        data/py.cc
        data/record_reader.cc
        data/round_robin_data_source.cc
        data/sample_data_source.cc
        data/shard_data_source.cc
        data/shuffle_data_source.cc
        data/skip_data_source.cc
        data/take_data_source.cc
        data/tape.cc
        data/yield_from_data_source.cc
        data/zip_data_source.cc
        data/zip_file_data_source.cc
        data/audio/audio_decoder.cc
        data/audio/waveform_to_fbank_converter.cc
        data/audio/detail/kaldi_fbank.cc
        data/audio/detail/sndfile.cc
        data/detail/file.cc
        data/detail/file_system.cc
        data/image/image_decoder.cc
        data/text/string_splitter.cc
        data/text/string_to_int_converter.cc
        data/text/string_to_tensor_converter.cc
        data/text/text_data_source.cc
        data/text/text_line_reader.cc
        data/text/text_reader.cc
        data/text/utf8_stream.cc
        data/text/detail/utf.cc
        data/text/sentencepiece/sp_decoder.cc
        data/text/sentencepiece/sp_encoder.cc
        data/text/sentencepiece/sp_model.cc
        data/text/sentencepiece/sp_processor.cc
)

if(FAIRSEQ2N_SUPPORT_IMAGE)
    target_sources(fairseq2n
        PRIVATE
            data/image/detail/jpeg_decompress_struct.cc
            data/image/detail/png_read_struct.cc
    )
endif()

if(FAIRSEQ2N_USE_CUDA)
    target_sources(fairseq2n
        PRIVATE
            generation/ngram_repeat_block.cu
    )
endif()

fairseq2n_set_compile_options(fairseq2n)

target_compile_features(fairseq2n PUBLIC cxx_std_17)

if(FAIRSEQ2N_SUPPORT_IMAGE)
    target_compile_definitions(fairseq2n PRIVATE FAIRSEQ2N_SUPPORT_IMAGE)
endif()

if(FAIRSEQ2N_USE_CUDA)
    target_compile_features(fairseq2n PRIVATE cuda_std_17)

    target_compile_definitions(fairseq2n PRIVATE FAIRSEQ2N_USE_CUDA)
endif()

if(FAIRSEQ2N_THREAD_LIB STREQUAL "tbb")
    target_compile_definitions(fairseq2n PRIVATE FAIRSEQ2N_USE_TBB)
endif()

if(PROJECT_IS_TOP_LEVEL)
    set(system)
else()
    set(system SYSTEM)
endif()

target_include_directories(fairseq2n ${system}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
)

target_link_libraries(fairseq2n
    PRIVATE
        ${CMAKE_DL_LIBS}
    PRIVATE
        fmt::fmt
        Iconv::Iconv
        kaldi-native-fbank::core
        kuba-zip
        natsort
        Threads::Threads
        sentencepiece-static
        SndFile::sndfile
    PUBLIC
        torch
)

if(FAIRSEQ2N_SUPPORT_IMAGE)
    target_link_libraries(fairseq2n PRIVATE jpeg_turbo_static png_static)
endif()

if(FAIRSEQ2N_USE_CUDA)
    target_link_libraries(fairseq2n PRIVATE CUDA::cudart)
endif()

if(FAIRSEQ2N_THREAD_LIB STREQUAL "tbb")
    target_link_libraries(fairseq2n PRIVATE TBB::tbb)
endif()

fairseq2n_set_link_options(fairseq2n)

set_target_properties(fairseq2n PROPERTIES
    VERSION
        ${PROJECT_VERSION}
    SOVERSION
        ${PROJECT_VERSION_MAJOR}
)

add_library(fairseq2n::fairseq2n ALIAS fairseq2n)

install(
    TARGETS
        fairseq2n
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
)

# ------------------------------------------------------------
# Library Configuration
# ------------------------------------------------------------

if(FAIRSEQ2N_SUPPORT_IMAGE)
    set(SUPPORTS_IMAGE "true")
else()
    set(SUPPORTS_IMAGE "false")
endif()

if(FAIRSEQ2N_USE_CUDA)
    set(USES_CUDA "true")

    set(CUDA_VERSION_MAJOR "${CUDAToolkit_VERSION_MAJOR}")
    set(CUDA_VERSION_MINOR "${CUDAToolkit_VERSION_MINOR}")
else()
    set(USES_CUDA "false")

    set(CUDA_VERSION_MAJOR "std::nullopt")
    set(CUDA_VERSION_MINOR "std::nullopt")
endif()

configure_file(config.h.in config.h @ONLY)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/config.h
    DESTINATION
        ${install_inc_dir}/fairseq2n
    COMPONENT
        devel
)
