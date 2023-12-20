# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(FindPackageHandleStandardArgs)

macro(__torch_determine_version)
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
        message(FATAL_ERROR "fairseq2n cannot determine PEP 440 version of PyTorch!")
    endif()

    if(TORCH_PEP440_VERSION MATCHES "^[0-9]+\.[0-9]+(\.[0-9]+)?")
        set(TORCH_VERSION ${CMAKE_MATCH_0})
    endif()

    unset(result)
endmacro()

macro(__torch_determine_cuda_version)
    execute_process(
        COMMAND
            ${Python3_EXECUTABLE} -c "import torch; print(torch.version.cuda or '')"
        OUTPUT_VARIABLE
            TORCH_CUDA_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE
            result
    )

    if(result GREATER 0)
        message(FATAL_ERROR "fairseq2n cannot determine CUDA version of PyTorch!")
    endif()

    # We ignore the patch since it is not relevant for compatibility checks.
    if(TORCH_CUDA_VERSION MATCHES "^([0-9]+)\.([0-9]+)")
        set(TORCH_CUDA_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(TORCH_CUDA_VERSION_MINOR ${CMAKE_MATCH_2})
    endif()

    unset(result)
endmacro()

if(FAIRSEQ2_USE_LIBTORCH)
    # TODO(balioglu): support libtorch
    message(FATAL_ERROR
        "`libtorch` is not supported yet."
    )
endif()

find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND
            ${Python3_EXECUTABLE} -c "import torch; print(torch.__file__)"
        OUTPUT_VARIABLE
            torch_init_file
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    cmake_path(CONVERT ${torch_init_file} TO_CMAKE_PATH_LIST torch_init_file NORMALIZE)

    cmake_path(REPLACE_FILENAME torch_init_file lib OUTPUT_VARIABLE torch_lib_dir)
    cmake_path(REPLACE_FILENAME torch_init_file include OUTPUT_VARIABLE torch_include_dir)

    cmake_path(
        APPEND
            torch_include_dir
        #INPUTS
            torch csrc api include
        OUTPUT_VARIABLE
            torch_api_include_dir
    )

    unset(torch_init_file)
endif()

find_library(TORCH_LIBRARY torch HINTS ${torch_lib_dir} NO_DEFAULT_PATH)
find_library(TORCH_CPU_LIBRARY torch_cpu HINTS ${torch_lib_dir} NO_DEFAULT_PATH)
find_library(TORCH_CUDA_LIBRARY torch_cuda HINTS ${torch_lib_dir} NO_DEFAULT_PATH)

find_library(C10_LIBRARY c10 HINTS ${torch_lib_dir} NO_DEFAULT_PATH)
find_library(C10_CUDA_LIBRARY c10_cuda HINTS ${torch_lib_dir} NO_DEFAULT_PATH)

find_library(TORCH_PYTHON_LIBRARY torch_python HINTS ${torch_lib_dir} NO_DEFAULT_PATH)

find_path(TORCH_INCLUDE_DIR torch HINTS ${torch_include_dir} NO_DEFAULT_PATH)
find_path(TORCH_API_INCLUDE_DIR torch HINTS ${torch_api_include_dir} NO_DEFAULT_PATH)

set(torch_required_vars
    TORCH_LIBRARY
    TORCH_CPU_LIBRARY
    C10_LIBRARY
    TORCH_INCLUDE_DIR
    TORCH_API_INCLUDE_DIR
)

mark_as_advanced(${torch_required_vars} TORCH_CUDA_LIBRARY C10_CUDA_LIBRARY TORCH_PYTHON_LIBRARY)

if(TORCH_LIBRARY)
    __torch_determine_version()
endif()

find_package_handle_standard_args(Torch
    REQUIRED_VARS
        ${torch_required_vars}
    VERSION_VAR
        TORCH_VERSION
)

unset(torch_lib_dir)
unset(torch_include_dir)
unset(torch_api_include_dir)
unset(torch_required_vars)

if(NOT Torch_FOUND)
    return()
endif()

if(TORCH_CUDA_LIBRARY)
    __torch_determine_cuda_version()

    set(TORCH_VARIANT "CUDA ${TORCH_CUDA_VERSION_MAJOR}.${TORCH_CUDA_VERSION_MINOR}")
else()
    set(TORCH_VARIANT "CPU-only")
endif()

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
            RESULT_VARIABLE
                result
        )

        if(result EQUAL 0)
            target_compile_definitions(torch_cxx11_abi
                INTERFACE
                    _GLIBCXX_USE_CXX11_ABI=$<BOOL:${TORCH_CXX11_ABI}>
            )
        endif()

        unset(result)
    endif()
endif()

if(NOT TARGET torch)
    add_library(torch SHARED IMPORTED)

    set_property(TARGET torch PROPERTY IMPORTED_LOCATION ${TORCH_LIBRARY})

    target_include_directories(torch INTERFACE ${TORCH_INCLUDE_DIR} ${TORCH_API_INCLUDE_DIR})

    target_link_libraries(torch INTERFACE ${TORCH_CPU_LIBRARY} ${C10_LIBRARY} torch_cxx11_abi)

    if(TORCH_CUDA_LIBRARY AND C10_CUDA_LIBRARY)
        target_link_libraries(torch INTERFACE ${TORCH_CUDA_LIBRARY} ${C10_CUDA_LIBRARY})
    endif()
endif()

if(NOT TARGET torch_python)
    add_library(torch_python SHARED IMPORTED)

    set_property(TARGET torch_python PROPERTY IMPORTED_LOCATION ${TORCH_PYTHON_LIBRARY})

    target_link_libraries(torch_python INTERFACE torch)
endif()
