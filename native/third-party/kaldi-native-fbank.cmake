# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

macro(fairseq2n_add_kaldi_native_fbank)
    if(NOT TARGET kaldi-native-fbank::core)
        find_package(Git REQUIRED)

        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} apply ${PROJECT_SOURCE_DIR}/third-party/kaldi-native-fbank.patch
            WORKING_DIRECTORY
                ${PROJECT_SOURCE_DIR}/third-party/kaldi-native-fbank
            ERROR_QUIET
        )

        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        set(KALDI_NATIVE_FBANK_BUILD_TESTS  OFF)
        set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF)

        set(backup_build_shared_libs ${BUILD_SHARED_LIBS})

        # Force the library to be static.
        set(BUILD_SHARED_LIBS FALSE)

        add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/kaldi-native-fbank EXCLUDE_FROM_ALL)

        # Revert.
        set(BUILD_SHARED_LIBS ${backup_build_shared_libs})

        unset(backup_build_shared_libs)

        set_target_properties(kaldi-native-fbank-core PROPERTIES
            CXX_VISIBILITY_PRESET
                hidden
            POSITION_INDEPENDENT_CODE
                ON
        )

        target_include_directories(kaldi-native-fbank-core SYSTEM
            PUBLIC
                ${PROJECT_SOURCE_DIR}/third-party/kaldi-native-fbank
        )

        # We depend on the phony torch_cxx11_abi target to ensure that we use
        # the same libstdc++ ABI as PyTorch.
        target_link_libraries(kaldi-native-fbank-core PRIVATE torch_cxx11_abi)

        add_library(kaldi-native-fbank::core ALIAS kaldi-native-fbank-core)
    endif()
endmacro()
