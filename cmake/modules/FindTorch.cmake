# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(FindPackageHandleStandardArgs)

macro(__torch_determine_cuda_version)
    execute_process(
        COMMAND
            ${Python3_EXECUTABLE} -c "import torch; print(torch.version.cuda or '')"
        OUTPUT_VARIABLE
            cuda_version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE
            result
    )

    if(NOT result EQUAL 0)
        message(FATAL_ERROR "fairseq2 cannot determine the version of CUDA used by PyTorch!")
    endif()

    # We ignore the patch since it is not relevant for compatibility checks.
    if(cuda_version MATCHES "^([0-9]+)\.([0-9]+)")
        set(TORCH_CUDA_VERSION       ${CMAKE_MATCH_0})
        set(TORCH_CUDA_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(TORCH_CUDA_VERSION_MINOR ${CMAKE_MATCH_2})
    endif()

    unset(result)
    unset(cuda_version)
endmacro()

execute_process(
    COMMAND
        ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE
        torch_cmake_prefix_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_FAKE_CUDNN)
    # Torch CMake package superficially has a hard dependency on cuDNN. As a
    # workaround, we override the cuDNN CMake variables with some fake paths.
    set(CUDNN_INCLUDE_PATH ${PROJECT_BINARY_DIR}/third-party/cudnn)
    set(CUDNN_LIBRARY_PATH ${PROJECT_BINARY_DIR}/third-party/cudnn/libcudnn.so)

    file(WRITE ${CUDNN_INCLUDE_PATH}/cudnn_version.h
[=[
#define CUDNN_MAJOR 7
#define CUDNN_MINOR 0
#define CUDNN_PATCH 0
]=])
endif()

find_package(Torch CONFIG PATHS ${torch_cmake_prefix_path})

if(TORCH_FAKE_CUDNN)
    # Since we don't really have cuDNN, we have to ensure that CMake does not
    # attempt to link against it.
    if(TARGET caffe2::cudnn-public)
        set_property(TARGET caffe2::cudnn-public PROPERTY INTERFACE_LINK_LIBRARIES "")
    endif()
endif()

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${TORCH_INSTALL_PREFIX}/lib NO_DEFAULT_PATH)

mark_as_advanced(TORCH_PYTHON_LIBRARY)

unset(torch_lib_dir)
unset(torch_cmake_prefix_path)

find_package_handle_standard_args(Torch
    REQUIRED_VARS
        TORCH_LIBRARY TORCH_PYTHON_LIBRARY
    VERSION_VAR
        Torch_VERSION
)

if(Torch_FOUND)
    if(NOT TARGET torch_cxx11_abi)
        # This is a phony target that propagates the libstdc++ ABI used by PyTorch.
        add_library(torch_cxx11_abi INTERFACE IMPORTED)

        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            execute_process(
                COMMAND
                    ${Python3_EXECUTABLE} -c "import torch; print(torch.compiled_with_cxx11_abi())"
                OUTPUT_VARIABLE
                    TORCH_CXX11_ABI
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )

            if(DEFINED TORCH_CXX11_ABI)
                target_compile_definitions(torch_cxx11_abi INTERFACE
                    _GLIBCXX_USE_CXX11_ABI=$<BOOL:${TORCH_CXX11_ABI}>
                )
            endif()
        endif()
    endif()

    if(NOT TARGET torch_python)
        execute_process(
            COMMAND
                ${Python3_EXECUTABLE} -c "import torch; print(torch.__version__)"
            OUTPUT_VARIABLE
                TORCH_PEP440_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE
                result
        )

        if(result GREATER 0)
            message(FATAL_ERROR "fairseq2 cannot determine the PEP 440 version of PyTorch!")
        endif()

        unset(result)

        add_library(torch_python SHARED IMPORTED)

        set_target_properties(torch_python PROPERTIES IMPORTED_LOCATION ${TORCH_PYTHON_LIBRARY})
    endif()

    __torch_determine_cuda_version()
endif()
