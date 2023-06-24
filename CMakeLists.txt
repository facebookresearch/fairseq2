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

include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(CheckLanguage)
include(GNUInstallDirs)

if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

include(cmake/base.cmake)
include(cmake/summary.cmake)

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------

option(FAIRSEQ2_BUILD_FOR_NATIVE
    #DESCRIPTION
        "Builds for the processor type of the compiling machine."
    #VALUE
        OFF
)
option(FAIRSEQ2_EDITABLE_PYTHON
    #DESCRIPTION
        "Copies Python extension module to the source tree for `pip install --editable`."
    #VALUE
        ON
)
option(FAIRSEQ2_INSTALL_STANDALONE
    #DESCRIPTION
        "Installs with relative rpath to the lib directory."
    #VALUE
        ON
)
cmake_dependent_option(FAIRSEQ2_PYENV
    #DESCRIPTION
        "Installs with relative rpath to the lib directory of the Python virtual environment."
    #VALUE
        ON
    #DEPENDS
        FAIRSEQ2_INSTALL_STANDALONE
    #HIDDEN_VALUE
        OFF
)
option(FAIRSEQ2_PERFORM_LTO
    #DESCRIPTION
        "Performs link-time optimization."
    #VALUE
        OFF
)
option(FAIRSEQ2_RUN_CLANG_TIDY
    #DESCRIPTION
        "Runs clang-tidy as static analyzer during compilation."
    #VALUE
        OFF
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

option(FAIRSEQ2_TREAT_WARNINGS_AS_ERRORS
    #DESCRIPTION
        "Treats compilation warnings as errors."
    #VALUE
        OFF
)
option(FAIRSEQ2_USE_CUDA
    #DESCRIPTION
        "Builds the CUDA kernels."
    #VALUE
        OFF
)

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------

# Some dependencies transitively use the pthreads API. This option ensures that
# they use the -pthread flag.
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
if(Python3_VERSION VERSION_LESS 3.8)
    message(FATAL_ERROR "fairseq2 requires CPython 3.8 or greater!")
endif()

find_package(Iconv REQUIRED)

find_package(Threads REQUIRED)

find_package(TBB 2021 REQUIRED)

find_package(Torch 1.11 REQUIRED)

if(FAIRSEQ2_RUN_CLANG_TIDY)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "fairseq2 requires Clang when `FAIRSEQ2_RUN_CLANG_TIDY` is set!")
    endif()

    find_package(ClangTidy REQUIRED)
endif()

add_subdirectory(third-party)

fairseq2_add_fmt()

fairseq2_add_natsort()

fairseq2_add_pybind11()

fairseq2_add_sentencepiece()

fairseq2_add_zip()

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
    fairseq2_add_gtest()
endif()

# ------------------------------------------------------------
# CUDA
# ------------------------------------------------------------

# By default, we build our CUDA kernels only for the Volta architecture.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70-real 70-virtual)
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
            "fairseq2 will use CUDA Toolkit ${CUDAToolkit_VERSION} which has a minor version mismatch with CUDA ${TORCH_CUDA_VERSION} that was used to build PyTorch. Most likely this should not be a problem."
        )
    endif()
endif()

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------

if(FAIRSEQ2_INSTALL_STANDALONE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(rpath_origin @loader_path)
    else()
        set(rpath_origin \$ORIGIN)
    endif()

    set(install_lib_dir fairseq2/lib)
    set(install_include_dir fairseq2/include)
else()
    set(install_lib_dir ${CMAKE_INSTALL_LIBDIR})
    set(install_include_dir ${CMAKE_INSTALL_INCLUDEDIR})
endif()

fairseq2_set_sanitizers()

add_subdirectory(src/fairseq2/native)

if(PROJECT_IS_TOP_LEVEL)
    add_subdirectory(src/fairseq2/native/extension)

    if(BUILD_TESTING)
        add_subdirectory(tests/native)
    endif()
endif()

# ------------------------------------------------------------
# Package Installation
# ------------------------------------------------------------

set(install_pkg_dir ${install_lib_dir}/cmake/fairseq2-${PROJECT_VERSION})

configure_package_config_file(
    #INPUT
        ${PROJECT_SOURCE_DIR}/cmake/package-config.cmake.in
    #OUTPUT
        ${PROJECT_BINARY_DIR}/lib/cmake/fairseq2/fairseq2-config.cmake
    INSTALL_DESTINATION
        ${install_pkg_dir}
    NO_SET_AND_CHECK_MACRO
)

write_basic_package_version_file(
    #OUTPUT
        ${PROJECT_BINARY_DIR}/lib/cmake/fairseq2/fairseq2-config-version.cmake
    VERSION
        ${PROJECT_VERSION}
    COMPATIBILITY
        AnyNewerVersion
)

install(
    FILES
        ${PROJECT_BINARY_DIR}/lib/cmake/fairseq2/fairseq2-config.cmake
        ${PROJECT_BINARY_DIR}/lib/cmake/fairseq2/fairseq2-config-version.cmake
    DESTINATION
        ${install_pkg_dir}
    COMPONENT
        devel
)

install(
    EXPORT
        fairseq2-targets
    FILE
        fairseq2-targets.cmake
    DESTINATION
        ${install_pkg_dir}
    COMPONENT
        devel
    NAMESPACE
        fairseq2::
)

export(
    EXPORT
        fairseq2-targets
    FILE
        ${PROJECT_BINARY_DIR}/lib/cmake/fairseq2/fairseq2-targets.cmake
    NAMESPACE
        fairseq2::
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

fairseq2_print_project_summary()
