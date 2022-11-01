# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include_guard(DIRECTORY)

function(fairseq2_add_third_party)
    foreach(project IN ITEMS ${ARGV})
        add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/${project} EXCLUDE_FROM_ALL)
    endforeach()
endfunction()

function(fairseq2_find_package package version)
    set(source_var FAIRSEQ2_${package}_SOURCE)

    # If the user has not specified an individual source for the package, use
    # the global setting.
    if(NOT DEFINED ${source_var})
        set(source_var FAIRSEQ2_DEPENDENCY_SOURCE)
    endif()

    set(source ${${source_var}})

    # If the source is `SYSTEM`, try to find the package only in system paths
    # using `find_package()`.
    if(source STREQUAL "SYSTEM")
        find_package(${package} ${version} REQUIRED)

        return()
    endif()

    # If the source is `AUTO`, try to find the package in system paths similar
    # to `SYSTEM`, but fall back to the third-party directory if it fails.
    if(source STREQUAL "AUTO")
        find_package(${package} ${version} QUIET)

        if(${package}_FOUND)
            return()
        endif()

        set(source BUNDLED)
    endif()

    # Lastly, if the source is `BUNDLED`, do not check the system paths and
    # directly use the source code in the third-party directory.
    if(source STREQUAL "BUNDLED")
        fairseq2_add_third_party(${package})

        return()
    endif()

    message(FATAL_ERROR "`${source_var}` must be `AUTO`, `SYSTEM`, or `BUNDLED`.")
endfunction()
