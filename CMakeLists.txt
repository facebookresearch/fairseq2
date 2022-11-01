# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.21.0)

project(fairseq2 VERSION 0.1.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE RelWithDebInfo)
endif()

include(CheckLanguage)

if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
endif()

include(cmake/base.cmake)
include(cmake/dependency.cmake)
include(cmake/fmt.cmake)
include(cmake/summary.cmake)
include(cmake/torch.cmake)

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------

option(FAIRSEQ2_BUILD_FOR_NATIVE
    #DESCRIPTION
        "Builds for the processor type of the compiling machine."
    #VALUE
        OFF
)
option(FAIRSEQ2_TREAT_WARNINGS_AS_ERRORS
    #DESCRIPTION
        "Treats compilation warnings as errors."
    #VALUE
        OFF
)
option(FAIRSEQ2_PERFORM_LTO
    #DESCRIPTION
        "Performs link-time optimization."
    #VALUE
        OFF
)
option(FAIRSEQ2_INSTALL_STANDALONE
    #DESCRIPTION
        "Installs with relative rpaths."
    #VALUE
        ON
)
option(FAIRSEQ2_RUN_CLANG_TIDY
    #DESCRIPTION
        "Runs clang-tidy as static analyzer during compilation."
    #VALUE
        OFF
)
option(FAIRSEQ2_EDITABLE_PYTHON
    #DESCRIPTION
        "Copies Python extension modules to the source tree for `pip install --editable`."
    #VALUE
        ON
)

set(FAIRSEQ2_DEPENDENCY_SOURCE
    #VALUE
        "BUNDLED"
    #TYPE
        CACHE STRING
    #DESCRIPTION
        "Instructs the build system on how to resolve the dependencies."
)
set_property(CACHE FAIRSEQ2_DEPENDENCY_SOURCE PROPERTY
    STRINGS
        "AUTO" "SYSTEM" "BUNDLED"
)

set(FAIRSEQ2_USE_CUDA
    #VALUE
        "AUTO"
    #TYPE
        CACHE STRING
    #DESCRIPTION
        "Uses CUDA. If not specified (i.e. AUTO), uses CUDA if PyTorch supports it."
)
set_property(CACHE FAIRSEQ2_USE_CUDA PROPERTY
    STRINGS
        "AUTO" "ON" "OFF"
)

set(FAIRSEQ2_SANITIZERS
    #VALUE
        ""
    #TYPE
        CACHE STRING
    #DESCRIPTION
        "Sanitizers to use. Supported values are 'asan', 'ubsan', and 'tsan'."
)
set_property(CACHE FAIRSEQ2_SANITIZERS PROPERTY
    STRINGS
        "" "asan" "ubsan" "tsan"
)

# By default, we build our CUDA kernels only for the Volta architecture.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70-real 70-virtual)
endif()

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

if(Python3_VERSION VERSION_LESS 3.8)
    message(FATAL_ERROR "fairseq2 requires CPython 3.8 or greater!")
endif()

fairseq2_find_torch(1.11)

fairseq2_find_fmt(9.1.0)

fairseq2_find_package(pybind11 2.10.0)

fairseq2_add_third_party(natsort)

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
    fairseq2_find_package(gtest)

    include(GoogleTest)
endif()

# ------------------------------------------------------------
# CUDA
# ------------------------------------------------------------

if(FAIRSEQ2_USE_CUDA STREQUAL "AUTO")
    if(TORCH_CUDA_VERSION)
        set(FAIRSEQ2_USE_CUDA ON)
    else()
        set(FAIRSEQ2_USE_CUDA OFF)
    endif()
endif()

if(FAIRSEQ2_USE_CUDA)
    if(NOT TORCH_CUDA_VERSION)
        message(FATAL_ERROR
            "fairseq2 requires a CUDA version of PyTorch when `FAIRSEQ2_USE_CUDA` is set!"
        )
    endif()

    enable_language(CUDA)

    find_package(CUDAToolkit REQUIRED)

    if(NOT CUDAToolkit_VERSION_MAJOR EQUAL TORCH_CUDA_VERSION_MAJOR)
        message(FATAL_ERROR
            "fairseq2 requires the version of the CUDA Toolkit (${CUDAToolkit_VERSION}) to match the version of CUDA that was used to build PyTorch (${TORCH_CUDA_VERSION})!"
        )
    endif()

    if(NOT CUDAToolkit_VERSION_MINOR EQUAL TORCH_CUDA_VERSION_MINOR)
        message(WARNING
            "fairseq2 will use the CUDA Toolkit ${CUDAToolkit_VERSION} which has a minor version mismatch with CUDA ${TORCH_CUDA_VERSION} that was used to build PyTorch. Most likely this should not be a problem."
        )
    endif()
endif()

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------

add_subdirectory(src/fairseq2)

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

# ------------------------------------------------------------
# Installation
# ------------------------------------------------------------

fairseq2_install_package()

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

fairseq2_print_configuration_summary()
