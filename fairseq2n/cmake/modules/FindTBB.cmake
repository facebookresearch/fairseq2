# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(FindPackageHandleStandardArgs)

find_package(TBB QUIET CONFIG)
if(TBB_FOUND)
    find_package_handle_standard_args(TBB CONFIG_MODE)

    return()
endif()

# The tbb PyPI package installs oneTBB under the lib directory of the Python
# environment. Check if we can find it there.
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND AND NOT TBB_LIBRARY)
    message(STATUS "Checking for oneTBB under the Python environment...")

    set(tbb_base_dir ${Python3_EXECUTABLE})

    cmake_path(GET tbb_base_dir PARENT_PATH tbb_base_dir)
    cmake_path(GET tbb_base_dir PARENT_PATH tbb_base_dir)

    cmake_path(APPEND tbb_base_dir lib OUTPUT_VARIABLE tbb_lib_dir)
    cmake_path(APPEND tbb_base_dir include OUTPUT_VARIABLE tbb_include_dir)

    unset(tbb_base_dir)
endif()

find_library(TBB_LIBRARY tbb HINTS ${tbb_lib_dir} NO_DEFAULT_PATH)
find_library(TBBMALLOC_LIBRARY tbbmalloc HINTS ${tbb_lib_dir} NO_DEFAULT_PATH)

find_path(TBB_INCLUDE_DIR tbb HINTS ${tbb_include_dir} NO_DEFAULT_PATH)

mark_as_advanced(TBB_LIBRARY TBBMALLOC_LIBRARY TBB_INCLUDE_DIR)

if(TBB_INCLUDE_DIR)
    set(TBB_VERSION 2021.8.0)  # TODO(balioglu): Infer this!
endif()

find_package_handle_standard_args(TBB
    REQUIRED_VARS
        TBB_LIBRARY TBBMALLOC_LIBRARY TBB_INCLUDE_DIR
    VERSION_VAR
        TBB_VERSION
)

unset(tbb_lib_dir)
unset(tbb_include_dir)

if(NOT TBB_FOUND)
    return()
endif()

if(NOT TARGET TBB::tbb)
    add_library(TBB::tbb SHARED IMPORTED)

    set_property(TARGET TBB::tbb PROPERTY IMPORTED_LOCATION ${TBB_LIBRARY})

    target_include_directories(TBB::tbb INTERFACE ${TBB_INCLUDE_DIR})
endif()

if(NOT TARGET TBB::tbbmalloc)
    add_library(TBB::tbbmalloc SHARED IMPORTED)

    set_property(TARGET TBB::tbbmalloc PROPERTY IMPORTED_LOCATION ${TBBMALLOC_LIBRARY})

    target_include_directories(TBB::tbbmalloc INTERFACE ${TBB_INCLUDE_DIR})
endif()
