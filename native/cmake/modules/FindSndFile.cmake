# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(FindPackageHandleStandardArgs)

find_package(SndFile QUIET CONFIG)
if(SndFile_FOUND)
    find_package_handle_standard_args(SndFile CONFIG_MODE)

    return()
endif()

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(SndFile QUIET sndfile)
endif()

find_library(SndFile_LIBRARY sndfile HINTS ${SndFile_LIBRARY_DIRS})

find_path(SndFile_INCLUDE_DIR sndfile.h HINTS ${SndFile_INCLUDE_DIRS})

mark_as_advanced(SndFile_LIBRARY SndFile_INCLUDE_DIR)

find_package_handle_standard_args(SndFile
    REQUIRED_VARS
        SndFile_LIBRARY SndFile_INCLUDE_DIR
    VERSION_VAR
        SndFile_VERSION
)

if(NOT SndFile_FOUND)
    return()
endif()

if(NOT TARGET SndFile::sndfile)
    add_library(SndFile::sndfile SHARED IMPORTED)

    set_property(TARGET SndFile::sndfile PROPERTY IMPORTED_LOCATION ${SndFile_LIBRARY})

    target_include_directories(SndFile::sndfile INTERFACE ${SndFile_INCLUDE_DIR})
endif()
