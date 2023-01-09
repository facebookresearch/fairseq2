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
        message(FATAL_ERROR "fairseq2 cannot determine the PEP 440 version of PyTorch!")
    endif()

    if(TORCH_PEP440_VERSION MATCHES "^[0-9]+\.[0-9]+\.[0-9]")
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
        message(FATAL_ERROR "fairseq2 cannot determine the CUDA version of PyTorch!")
    endif()

    # We ignore the patch since it is not relevant for compatibility checks.
    if(TORCH_CUDA_VERSION MATCHES "^([0-9]+)\.([0-9]+)")
        set(TORCH_CUDA_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(TORCH_CUDA_VERSION_MINOR ${CMAKE_MATCH_2})
    endif()

    unset(result)
endmacro()

execute_process(
    COMMAND
        ${Python3_EXECUTABLE} -c "import torch; print(torch.__file__)"
    OUTPUT_VARIABLE
        torch_init_file
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

cmake_path(REPLACE_FILENAME torch_init_file lib OUTPUT_VARIABLE torch_lib_dir)
cmake_path(REPLACE_FILENAME torch_init_file include OUTPUT_VARIABLE torch_include_dir)

find_library(TORCH_LIBRARY torch PATHS ${torch_lib_dir})

find_library(TORCH_CPU_LIBRARY torch_cpu PATHS ${torch_lib_dir})

find_library(TORCH_CUDA_LIBRARY torch_cuda PATHS ${torch_lib_dir})

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${torch_lib_dir})

find_library(C10_LIBRARY c10 PATHS ${torch_lib_dir})

find_library(C10_CUDA_LIBRARY c10_cuda PATHS ${torch_lib_dir})

find_path(TORCH_INCLUDE_DIR torch PATHS ${torch_include_dir})

mark_as_advanced(${torch_required_vars} TORCH_CUDA_LIBRARY C10_CUDA_LIBRARY)

set(torch_required_vars
    TORCH_LIBRARY TORCH_CPU_LIBRARY TORCH_PYTHON_LIBRARY C10_LIBRARY TORCH_INCLUDE_DIR
)

__torch_determine_version()

find_package_handle_standard_args(Torch
    REQUIRED_VARS
        ${torch_required_vars}
    VERSION_VAR
        TORCH_VERSION
)

unset(torch_include_dir)
unset(torch_init_file)
unset(torch_lib_dir)
unset(torch_required_vars)

if(Torch_FOUND)
    if(TORCH_CUDA_LIBRARY)
        __torch_determine_cuda_version()
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
                target_compile_definitions(torch_cxx11_abi INTERFACE
                    _GLIBCXX_USE_CXX11_ABI=$<BOOL:${TORCH_CXX11_ABI}>
                )
            endif()

            unset(result)
        endif()
    endif()

    if(NOT TARGET torch)
        add_library(torch SHARED IMPORTED)

        set_property(TARGET torch PROPERTY IMPORTED_LOCATION ${TORCH_LIBRARY})

        target_include_directories(torch INTERFACE ${TORCH_INCLUDE_DIR})

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
endif()
