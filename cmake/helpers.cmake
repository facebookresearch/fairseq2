# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include_guard(DIRECTORY)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

function(fairseq2_add_third_party)
    foreach(project IN ITEMS ${ARGV})
        add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/${project} EXCLUDE_FROM_ALL)
    endforeach()
endfunction()

function(fairseq2_add_target target)
    cmake_parse_arguments(arg
        #OPTIONS
            "EXECUTABLE;LIBRARY;SHARED_LIBRARY;STATIC_LIBRARY;PYTHON_MODULE"
        #KEYWORDS
            "OUTPUT_NAME;PYTHON_MODULE_LOCATION"
        #MULTI_VALUE_KEYWORDS
            ""
        #ARGUMENTS
            ${ARGN}
    )

    if(arg_EXECUTABLE)
        add_executable(${target})
    elseif(arg_PYTHON_MODULE)
        if(NOT COMMAND Python3_add_library)
            message(FATAL_ERROR "`Python3` CMake module must be loaded before calling `fairseq2_add_target()` when the target type is `PYTHON_MODULE`!")
        endif()

        cmake_path(APPEND CMAKE_CURRENT_SOURCE_DIR
            #INPUT
                ${arg_PYTHON_MODULE_LOCATION}
            OUTPUT_VARIABLE
                py_module_source_dir
        )
        cmake_path(APPEND CMAKE_CURRENT_BINARY_DIR
            #INPUT
                ${arg_PYTHON_MODULE_LOCATION}
            OUTPUT_VARIABLE
                py_module_binary_dir
        )

        Python3_add_library(${target} WITH_SOABI)
    else()
        if(arg_LIBRARY)
            set(lib_type)
        elseif(arg_SHARED_LIBRARY)
            set(lib_type SHARED)
        elseif(arg_STATIC_LIBRARY)
            set(lib_type STATIC)
        else()
            message(FATAL_ERROR "`fairseq2_add_target()` is called with an invalid target type!")
        endif()

        add_library(${target} ${lib_type})
    endif()

    __fairseq2_set_properties()

    __fairseq2_set_compile_options()

    __fairseq2_set_link_options()

    if(FAIRSEQ2_SANITIZERS)
        __fairseq2_set_sanitizers()
    endif()

    if(FAIRSEQ2_RUN_CLANG_TIDY)
        __fairseq2_set_clang_tidy()
    endif()

    if(NOT arg_PYTHON_MODULE)
        __fairseq2_install()
    else()
        __fairseq2_install_py_module()
    endif()
endfunction()

function(__fairseq2_set_properties)
    if(arg_EXECUTABLE)
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/bin
        )

        target_include_directories(${target}
            PRIVATE
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
                $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
        )
    elseif(arg_PYTHON_MODULE)
        set_target_properties(${target} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY
                ${py_module_binary_dir}
        )

        target_include_directories(${target}
            PRIVATE
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
                $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
        )
    else()
        set_target_properties(${target} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/lib
            ARCHIVE_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/lib
        )

        if(PROJECT_IS_TOP_LEVEL)
            set(system)
        else()
            set(system SYSTEM)
        endif()

        target_include_directories(${target} ${system}
            PUBLIC
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
                $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
        )
    endif()

    set_target_properties(${target} PROPERTIES
        C_EXTENSIONS
            OFF
        C_VISIBILITY_PRESET
            hidden
        CXX_EXTENSIONS
            OFF
        CXX_VISIBILITY_PRESET
            hidden
        CUDA_EXTENSIONS
            OFF
        CUDA_VISIBILITY_PRESET
            hidden
        CUDA_SEPARABLE_COMPILATION
            ON
        POSITION_INDEPENDENT_CODE
            ON
        EXPORT_COMPILE_COMMANDS
            ON
    )

    if(arg_SHARED_LIBRARY AND NOT FAIRSEQ2_INSTALL_STANDALONE)
        set_target_properties(${target} PROPERTIES
            VERSION
                ${PROJECT_VERSION}
            SOVERSION
                ${PROJECT_VERSION_MAJOR}
        )
    endif()

    if(arg_PYTHON_MODULE AND FAIRSEQ2_DEVELOP_PYTHON)
        # We have to use absolute rpaths so that the in-source copy of the Python
        # extension module can find its dependencies under the build tree.
        set_target_properties(${target} PROPERTIES BUILD_RPATH_USE_ORIGIN OFF)

        # Copy the Python extension module to the source tree for
        # `pip install --editable` to work.
        add_custom_target(${target}_devel ALL
            COMMAND
                ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:${target}>" "${py_module_source_dir}"
            VERBATIM
        )

        add_dependencies(${target}_devel ${target})
    endif()

    if(arg_OUTPUT_NAME)
        set_target_properties(${target} PROPERTIES OUTPUT_NAME ${arg_OUTPUT_NAME})
    endif()
endfunction()

function(__fairseq2_set_compile_options)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "fairseq2 supports only GCC and Clang toolchains!")
    endif()

    target_compile_options(${target}
        PRIVATE
            -fasynchronous-unwind-tables -fstack-protector-strong
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
            message(FATAL_ERROR "fairseq2 requires GCC 7 or greater!")
        endif()

        target_compile_options(${target}
            PRIVATE
                -Wall
                -Wcast-align
                -Wconversion
                -Wdouble-promotion
                -Wextra
                -Wfloat-equal
                -Wformat=2
                -Winit-self
                -Wlogical-op
                -Wno-unknown-pragmas
                -Wpointer-arith
                -Wshadow
                -Wsign-conversion
                -Wswitch-enum
                -Wunused
                $<$<COMPILE_LANGUAGE:CXX,CUDA>:-Wnon-virtual-dtor>
                $<$<COMPILE_LANGUAGE:CXX,CUDA>:-Woverloaded-virtual>
                $<$<COMPILE_LANGUAGE:CXX,CUDA>:-Wuseless-cast>
        )

        target_compile_definitions(${target} PRIVATE $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
    else()
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
            message(FATAL_ERROR "fairseq2 requires Clang 7 or greater!")
        endif()

        target_compile_options(${target}
            PRIVATE
                -fsized-deallocation
                -Weverything
                -Wno-c++98-compat
                -Wno-c++98-compat-pedantic
                -Wno-exit-time-destructors
                -Wno-extra-semi-stmt
                -Wno-global-constructors
                -Wno-missing-variable-declarations
                -Wno-old-style-cast
                -Wno-padded
                -Wno-reserved-id-macro
                -Wno-shadow-uncaptured-local
                -Wno-used-but-marked-unused
                -Wno-zero-as-null-pointer-constant
        )
    endif()

    if(FAIRSEQ2_TREAT_WARNINGS_AS_ERRORS)
        target_compile_options(${target} PRIVATE -Werror)
    endif()

    if(FAIRSEQ2_BUILD_FOR_NATIVE)
        target_compile_options(${target} PRIVATE -march=native -mtune=native)
    endif()

    target_compile_definitions(${target} PRIVATE $<$<NOT:$<CONFIG:Debug>>:_FORTIFY_SOURCE=2>)
endfunction()

function(__fairseq2_set_link_options)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(${target}
            PRIVATE
                LINKER:--as-needed
                LINKER:--build-id=sha1
                LINKER:-z,noexecstack
                LINKER:-z,now
                LINKER:-z,relro
        )

        if(NOT arg_PYTHON_MODULE)
            target_link_options(${target} PRIVATE LINKER:-z,defs)
        endif()

        if(FAIRSEQ2_TREAT_WARNINGS_AS_ERRORS)
            target_link_options(${target} PRIVATE LINKER:--fatal-warnings)
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        target_link_options(${target} PRIVATE LINKER:-bind_at_load)

        if(NOT arg_PYTHON_MODULE)
            target_link_options(${target} PRIVATE LINKER:-undefined,error)
        else()
            target_link_options(${target} PRIVATE LINKER:-undefined,dynamic_lookup)
        endif()

        if(FAIRSEQ2_TREAT_WARNINGS_AS_ERRORS)
            target_link_options(${target} PRIVATE LINKER:-fatal_warnings)
        endif()
    else()
        message(FATAL_ERROR "fairseq2 supports only Linux and macOS operating systems!")
    endif()

    if(FAIRSEQ2_PERFORM_LTO)
        set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)

        if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            __fairseq2_set_macos_lto_path()
        endif()
    endif()
endfunction()

# When performing ThinLTO on macOS, mach-o object files are generated under a
# temporary directory that gets deleted by the linker at the end of the build
# process. Thus tools such as dsymutil cannot access the DWARF info contained
# in those files. To ensure that the object files still exist after the build
# process, we have to set the `object_path_lto` linker option.
function(__fairseq2_set_macos_lto_path)
    if(arg_STATIC_LIBRARY OR (arg_LIBRARY AND NOT BUILD_SHARED_LIBS))
        return()
    endif()

    set(lto_dir ${CMAKE_CURRENT_BINARY_DIR}/lto.dir/${target}/${CMAKE_CFG_INTDIR})

    add_custom_command(
        TARGET
            ${target}
        PRE_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E make_directory "${lto_dir}"
        VERBATIM
    )

    target_link_options(${target} PRIVATE LINKER:-object_path_lto "${lto_dir}")
endfunction()

function(__fairseq2_set_sanitizers)
    foreach(sanitizer IN ITEMS ${FAIRSEQ2_SANITIZERS})
        if(sanitizer STREQUAL "nosan")
            continue()
        elseif(sanitizer STREQUAL "asan")
            if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                target_compile_definitions(${target} PRIVATE _GLIBCXX_SANITIZE_VECTOR)
            endif()

            list(APPEND sanitizer_opts -fsanitize=address)
        elseif(sanitizer STREQUAL "ubsan")
            list(APPEND sanitizer_opts -fsanitize=undefined)
        elseif(sanitizer STREQUAL "tsan")
            list(APPEND sanitizer_opts -fsanitize=thread)
        else()
            message(FATAL_ERROR "fairseq2 does not support the '${sanitizer}' sanitizer!")
        endif()
    endforeach()

    target_compile_options(${target} PRIVATE ${sanitizer_opts} -fno-omit-frame-pointer)

    target_link_options(${target} PRIVATE ${sanitizer_opts})
endfunction()

function(__fairseq2_set_clang_tidy)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "fairseq2 requires Clang when `FAIRSEQ2_RUN_CLANG_TIDY` is set!")
    endif()

    find_program(FAIRSEQ2_CLANG_TIDY_PROG REQUIRED NAMES clang-tidy)

    mark_as_advanced(FAIRSEQ2_CLANG_TIDY_PROG)

    set_target_properties(${target} PROPERTIES
        C_CLANG_TIDY
            ${FAIRSEQ2_CLANG_TIDY_PROG}
        CXX_CLANG_TIDY
            ${FAIRSEQ2_CLANG_TIDY_PROG}
        CUDA_CLANG_TIDY
            ${FAIRSEQ2_CLANG_TIDY_PROG}
    )
endfunction()

function(__fairseq2_install)
    if(FAIRSEQ2_INSTALL_STANDALONE)
        __fairseq2_set_install_rpath()

        set(install_bin_dir ${PROJECT_NAME}/bin)
        set(install_lib_dir ${PROJECT_NAME}/lib)
        set(install_inc_dir ${PROJECT_NAME}/include)
    else()
        set(install_bin_dir ${CMAKE_INSTALL_BINDIR})
        set(install_lib_dir ${CMAKE_INSTALL_LIBDIR})
        set(install_inc_dir ${CMAKE_INSTALL_INCLUDEDIR})
    endif()

    if(arg_EXECUTABLE)
        install(
            TARGETS
                ${target}
            EXPORT
                ${PROJECT_NAME}-targets
            RUNTIME
                DESTINATION
                    ${install_bin_dir}
                COMPONENT
                    runtime
        )
    else()
        install(
            TARGETS
                ${target}
            EXPORT
                ${PROJECT_NAME}-targets
            LIBRARY
                DESTINATION
                    ${install_lib_dir}
                COMPONENT
                    runtime
                NAMELINK_COMPONENT
                    devel
            ARCHIVE
                DESTINATION
                    ${install_lib_dir}
                COMPONENT
                    devel
            INCLUDES DESTINATION
                ${install_inc_dir}
        )

        cmake_path(RELATIVE_PATH CMAKE_CURRENT_SOURCE_DIR
            BASE_DIRECTORY
                ${PROJECT_SOURCE_DIR}/src
            OUTPUT_VARIABLE
                relative_inc_dir
        )

        install(
            DIRECTORY
                ${CMAKE_CURRENT_SOURCE_DIR}/
            DESTINATION
                ${install_inc_dir}/${relative_inc_dir}
            COMPONENT
                devel
            FILES_MATCHING
                PATTERN "*.h"
            PATTERN "private" EXCLUDE
            PATTERN "CMakeFiles" EXCLUDE
        )

        install(
            DIRECTORY
                ${CMAKE_CURRENT_BINARY_DIR}/
            DESTINATION
                ${install_inc_dir}/${relative_inc_dir}
            COMPONENT
                devel
            FILES_MATCHING
                PATTERN "*.h"
            PATTERN "private" EXCLUDE
            PATTERN "CMakeFiles" EXCLUDE
        )
    endif()
endfunction()

function(__fairseq2_install_py_module)
    if(FAIRSEQ2_INSTALL_STANDALONE)
        __fairseq2_set_install_rpath()
    endif()

    cmake_path(RELATIVE_PATH py_module_source_dir
        BASE_DIRECTORY
            ${PROJECT_SOURCE_DIR}/src
        OUTPUT_VARIABLE
            relative_py_module_dir
    )

    install(
        TARGETS
            ${target}
        LIBRARY
            DESTINATION
                ${relative_py_module_dir}
            COMPONENT
                python_modules
        EXCLUDE_FROM_ALL
    )
endfunction()

function(__fairseq2_set_install_rpath)
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(rpath_origin @loader_path)
    else()
        set(rpath_origin \$ORIGIN)
    endif()

    if(arg_PYTHON_MODULE)
        set(py_dist_dir ${PROJECT_SOURCE_DIR}/src)

        cmake_path(RELATIVE_PATH py_dist_dir
            BASE_DIRECTORY
                ${py_module_source_dir}
            OUTPUT_VARIABLE
                relative_py_dist_dir
        )

        # Ensure that the Python extension module can find the shared libraries
        # contained in the distribution.
        set(rpath ${relative_py_dist_dir}/${PROJECT_NAME}/lib)
    elseif(arg_EXECUTABLE)
        set(rpath ../lib)
    else()
        set(rpath .)
    endif()

    set_target_properties(${target} PROPERTIES INSTALL_RPATH ${rpath_origin}/${rpath})
endfunction()

function(fairseq2_install_package)
    set(pkg_dir ${PROJECT_BINARY_DIR}/lib/cmake/${PROJECT_NAME})

    if(FAIRSEQ2_INSTALL_STANDALONE)
        set(install_lib_dir ${PROJECT_NAME}/lib)
    else()
        set(install_lib_dir ${CMAKE_INSTALL_LIBDIR})
    endif()

    set(install_pkg_dir ${install_lib_dir}/cmake/${PROJECT_NAME}-${PROJECT_VERSION})

    configure_package_config_file(
        #INPUT
            ${PROJECT_SOURCE_DIR}/src/${PROJECT_NAME}-config.cmake.in
        #OUTPUT
            ${pkg_dir}/${PROJECT_NAME}-config.cmake
        INSTALL_DESTINATION
            ${install_pkg_dir}
        NO_SET_AND_CHECK_MACRO
    )

    write_basic_package_version_file(
        #OUTPUT
            ${pkg_dir}/${PROJECT_NAME}-config-version.cmake
        VERSION
            ${PROJECT_VERSION}
        COMPATIBILITY
            AnyNewerVersion
    )

    install(
        FILES
            ${pkg_dir}/${PROJECT_NAME}-config.cmake
            ${pkg_dir}/${PROJECT_NAME}-config-version.cmake
        DESTINATION
            ${install_pkg_dir}
        COMPONENT
            devel
    )

    install(
        EXPORT
            ${PROJECT_NAME}-targets
        FILE
            ${PROJECT_NAME}-targets.cmake
        DESTINATION
            ${install_pkg_dir}
        COMPONENT
            devel
        NAMESPACE
            ${PROJECT_NAME}::
    )

    export(
        EXPORT
            ${PROJECT_NAME}-targets
        FILE
            ${pkg_dir}/${PROJECT_NAME}-targets.cmake
        NAMESPACE
            ${PROJECT_NAME}::
    )
endfunction()
