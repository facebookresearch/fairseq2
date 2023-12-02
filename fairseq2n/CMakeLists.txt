# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.21.0)

project(fairseq2n VERSION 0.2.1 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE RelWithDebInfo)
endif()

if(DEFINED ENV{CONDA_PREFIX} AND NOT DEFINED ENV{CONDA_BUILD_SYSROOT})
    message(FATAL_ERROR
        "It looks like you are in a Conda environment, but the compiler packages are not installed. Please run `conda install -c conda-forge compilers` first."
    )
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

# Prefer pthread over other thread libraries.
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------

option(FAIRSEQ2N_BUILD_FOR_NATIVE
    #DESCRIPTION
        "Builds for the processor type of the compiling machine."
    #VALUE
        OFF
)

option(FAIRSEQ2N_INSTALL_STANDALONE
    #DESCRIPTION
        "Installs with relative rpaths."
    #VALUE
        ON
)

option(FAIRSEQ2N_PERFORM_LTO
    #DESCRIPTION
        "Performs link-time optimization."
    #VALUE
        OFF
)

option(FAIRSEQ2N_RUN_CLANG_TIDY
    #DESCRIPTION
        "Runs clang-tidy as static analyzer during compilation."
    #VALUE
        OFF
)

set(FAIRSEQ2N_SANITIZERS
    #VALUE
        ""
    #TYPE
        CACHE STRING
    #DESCRIPTION
        "Sanitizers to enable."
)
set_property(CACHE FAIRSEQ2N_SANITIZERS PROPERTY
    STRINGS
        "" "asan" "ubsan" "tsan"
)

option(FAIRSEQ2N_TREAT_WARNINGS_AS_ERRORS
    #DESCRIPTION
        "Treats compilation warnings as errors."
    #VALUE
        OFF
)

option(FAIRSEQ2N_USE_LIBTORCH
    #DESCRIPTION
        "Uses libtorch instead of PyTorch."
    #VALUE
        OFF
)

option(FAIRSEQ2N_USE_CUDA
    #DESCRIPTION
        "Builds the CUDA kernels."
    #VALUE
        OFF
)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(default_thread_lib tbb)
else()
    set(default_thread_lib)
endif()

set(FAIRSEQ2N_THREAD_LIB
    #VALUE
        ${default_thread_lib}
    #TYPE
        CACHE STRING
    #DESCRIPTION
        "Thread library to use."
)
set_property(CACHE FAIRSEQ2N_THREAD_LIB PROPERTY
    STRINGS
        "" "tbb"
)

if(FAIRSEQ2N_THREAD_LIB STREQUAL "tbb")
    if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        message(FATAL_ERROR "Intel oneTBB is only supported on x86-64 systems.")
    endif()
endif()

option(FAIRSEQ2N_BUILD_PYTHON_BINDINGS
    #DESCRIPTION
        "Builds the Python bindings."
    #VALUE
        ON
)

cmake_dependent_option(FAIRSEQ2N_PYTHON_DEVEL
    #DESCRIPTION
        "Copies the Python extension module to the source tree for `pip install --editable`."
    #VALUE
        ON
    #DEPENDS_ON
        FAIRSEQ2N_BUILD_PYTHON_BINDINGS
    #HIDDEN_VALUE
        OFF
)

# ------------------------------------------------------------
# Sanitizers
# ------------------------------------------------------------

fairseq2n_set_sanitizers()

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------

find_package(Iconv REQUIRED)

find_package(SndFile 1.0.25 REQUIRED)

find_package(Threads REQUIRED)

if(FAIRSEQ2N_THREAD_LIB STREQUAL "tbb")
    find_package(TBB 2021.8 REQUIRED)
endif()

find_package(Torch 1.13 REQUIRED)

if(FAIRSEQ2N_BUILD_PYTHON_BINDINGS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    if(Python3_VERSION VERSION_LESS 3.8)
        message(FATAL_ERROR "fairseq2n requires CPython 3.8 or greater!")
    endif()
endif()

if(FAIRSEQ2N_RUN_CLANG_TIDY)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "fairseq2n requires Clang when `FAIRSEQ2N_RUN_CLANG_TIDY` is set!")
    endif()

    find_package(ClangTidy REQUIRED)
endif()

add_subdirectory(third-party)

fairseq2n_add_fmt()

fairseq2n_add_kaldi_native_fbank()

fairseq2n_add_natsort()

fairseq2n_add_sentencepiece()

fairseq2n_add_zip()

if(FAIRSEQ2N_BUILD_PYTHON_BINDINGS)
    fairseq2n_add_pybind11()
endif()

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
    fairseq2n_add_gtest()
endif()

# ------------------------------------------------------------
# CUDA
# ------------------------------------------------------------

# By default, we build our CUDA kernels only for the Volta architecture.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70-real 70-virtual)
endif()

if(FAIRSEQ2N_USE_CUDA)
    if(NOT TORCH_CUDA_VERSION)
        message(FATAL_ERROR
            "fairseq2n requires a CUDA version of PyTorch when `FAIRSEQ2N_USE_CUDA` is set!"
        )
    endif()

    enable_language(CUDA)

    find_package(CUDAToolkit REQUIRED)

    if(NOT CUDAToolkit_VERSION_MAJOR EQUAL TORCH_CUDA_VERSION_MAJOR)
        message(FATAL_ERROR
            "fairseq2n requires the version of the CUDA Toolkit (${CUDAToolkit_VERSION}) to match the version of CUDA that was used to build PyTorch (${TORCH_CUDA_VERSION})!"
        )
    endif()

    if(NOT CUDAToolkit_VERSION_MINOR EQUAL TORCH_CUDA_VERSION_MINOR)
        message(WARNING
            "fairseq2n will use CUDA Toolkit ${CUDAToolkit_VERSION} which has a minor version mismatch with CUDA ${TORCH_CUDA_VERSION} that was used to build PyTorch. Most likely this should not be a problem."
        )
    endif()
endif()

# ------------------------------------------------------------
# Package Installation
# ------------------------------------------------------------

if(FAIRSEQ2N_INSTALL_STANDALONE)
    set(install_lib_dir lib)
    set(install_inc_dir include)
else()
    set(install_lib_dir ${CMAKE_INSTALL_LIBDIR})
    set(install_inc_dir ${CMAKE_INSTALL_INCLUDEDIR})
endif()

set(install_pkg_dir ${install_lib_dir}/cmake/fairseq2n-${PROJECT_VERSION})

configure_package_config_file(
    #INPUT
        ${PROJECT_SOURCE_DIR}/src/fairseq2n-config.cmake.in
    #OUTPUT
        ${PROJECT_BINARY_DIR}/lib/cmake/fairseq2n/fairseq2n-config.cmake
    INSTALL_DESTINATION
        ${install_pkg_dir}
    NO_SET_AND_CHECK_MACRO
)

write_basic_package_version_file(
    #OUTPUT
        ${PROJECT_BINARY_DIR}/lib/cmake/fairseq2n/fairseq2n-config-version.cmake
    VERSION
        ${PROJECT_VERSION}
    COMPATIBILITY
        AnyNewerVersion
)

install(
    FILES
        ${PROJECT_BINARY_DIR}/lib/cmake/fairseq2n/fairseq2n-config.cmake
        ${PROJECT_BINARY_DIR}/lib/cmake/fairseq2n/fairseq2n-config-version.cmake
    DESTINATION
        ${install_pkg_dir}
    COMPONENT
        devel
)

install(
    EXPORT
        fairseq2n-targets
    FILE
        fairseq2n-targets.cmake
    DESTINATION
        ${install_pkg_dir}
    COMPONENT
        devel
    NAMESPACE
        fairseq2n::
)

export(
    EXPORT
        fairseq2n-targets
    FILE
        ${PROJECT_BINARY_DIR}/lib/cmake/fairseq2n/fairseq2n-targets.cmake
    NAMESPACE
        fairseq2n::
)

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------

add_subdirectory(src/fairseq2n)

if(FAIRSEQ2N_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python/src/fairseq2n/bindings)
endif()

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

fairseq2n_print_project_summary()
