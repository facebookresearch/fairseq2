# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include_guard(DIRECTORY)

function(fairseq2_find_torch version)
    execute_process(
        COMMAND
            ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE
            torch_cmake_prefix_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL
            ANY
    )

    find_package(Torch ${version} REQUIRED PATHS ${torch_cmake_prefix_path})

    __fairseq2_find_torch_python()

    __fairseq2_determine_cxx11_abi()

    __fairseq2_determine_cuda_version()

    __fairseq2_determine_pep440_version()

    set(TORCH_VERSION ${Torch_VERSION} PARENT_SCOPE)

    set(TORCH_CUDA_VERSION       ${TORCH_CUDA_VERSION}       PARENT_SCOPE)
    set(TORCH_CUDA_VERSION_MAJOR ${TORCH_CUDA_VERSION_MAJOR} PARENT_SCOPE)
    set(TORCH_CUDA_VERSION_MINOR ${TORCH_CUDA_VERSION_MINOR} PARENT_SCOPE)

    set(TORCH_PEP440_VERSION ${TORCH_PEP440_VERSION} PARENT_SCOPE)
endfunction()

function(__fairseq2_find_torch_python)
    cmake_path(GET TORCH_LIBRARY PARENT_PATH torch_library_dir)

    # Torch CMake package does not export torch_python; therefore we have to
    # explicitly find it.
    find_library(FAIRSEQ2_TORCH_PYTHON_LIBRARY torch_python REQUIRED PATHS ${torch_library_dir})

    mark_as_advanced(FAIRSEQ2_TORCH_PYTHON_LIBRARY)

    add_library(torch_python INTERFACE)

    target_link_libraries(torch_python INTERFACE ${FAIRSEQ2_TORCH_PYTHON_LIBRARY})
endfunction()

function(__fairseq2_determine_cxx11_abi)
    # This is a phony target that propagates the libstdc++ ABI used by PyTorch.
    add_library(torch_cxx11_abi INTERFACE)

    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        return()
    endif()

    execute_process(
        COMMAND
            ${Python3_EXECUTABLE} -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
        OUTPUT_VARIABLE
            use_cxx11_abi
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL
            ANY
    )

    target_compile_definitions(torch_cxx11_abi INTERFACE
        _GLIBCXX_USE_CXX11_ABI=$<BOOL:${use_cxx11_abi}>
    )
endfunction()

function(__fairseq2_determine_cuda_version)
    execute_process(
        COMMAND
            ${Python3_EXECUTABLE} -c "import torch; print(torch.version.cuda or '')"
        OUTPUT_VARIABLE
            cuda_version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL
            ANY
    )

    # We ignore the patch since it is not relevant for compatibility checks.
    if(cuda_version MATCHES "^([0-9]+)\.([0-9]+)")
        set(TORCH_CUDA_VERSION       ${CMAKE_MATCH_0} PARENT_SCOPE)
        set(TORCH_CUDA_VERSION_MAJOR ${CMAKE_MATCH_1} PARENT_SCOPE)
        set(TORCH_CUDA_VERSION_MINOR ${CMAKE_MATCH_2} PARENT_SCOPE)
    endif()
endfunction()

function(__fairseq2_determine_pep440_version)
    execute_process(
        COMMAND
            ${Python3_EXECUTABLE} -c "import torch; print(torch.__version__)"
        OUTPUT_VARIABLE
            pep440_version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL
            ANY
    )

    set(TORCH_PEP440_VERSION ${pep440_version} PARENT_SCOPE)
endfunction()
