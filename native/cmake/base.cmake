# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include_guard(DIRECTORY)

function(fairseq2n_set_compile_options target)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "fairseq2n supports only GCC and Clang toolchains!")
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

    if(FAIRSEQ2N_RUN_CLANG_TIDY)
        set_target_properties(${target} PROPERTIES
            C_CLANG_TIDY
                ${CLANG_TIDY_EXECUTABLE}
            CXX_CLANG_TIDY
                ${CLANG_TIDY_EXECUTABLE}
            CUDA_CLANG_TIDY
                ${CLANG_TIDY_EXECUTABLE}
        )
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
            message(FATAL_ERROR "fairseq2n requires GCC 7 or greater!")
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
                -Wshadow=compatible-local
                -Wsign-conversion
                -Wswitch
                -Wunused
                $<$<COMPILE_LANGUAGE:CXX,CUDA>:-Wnon-virtual-dtor>
                $<$<COMPILE_LANGUAGE:CXX,CUDA>:-Woverloaded-virtual>
                $<$<COMPILE_LANGUAGE:CXX,CUDA>:-Wuseless-cast>
        )

        target_compile_definitions(${target} PRIVATE $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
    else()
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12)
            message(FATAL_ERROR "fairseq2n requires Clang 12 or greater!")
        endif()

        target_compile_options(${target}
            PRIVATE
                -fsized-deallocation
                -Weverything
                -Wno-c++98-compat
                -Wno-c++98-compat-pedantic
                -Wno-disabled-macro-expansion
                -Wno-exit-time-destructors
                -Wno-extra-semi-stmt
                -Wno-global-constructors
                -Wno-macro-redefined
                -Wno-missing-variable-declarations
                -Wno-old-style-cast
                -Wno-padded
                -Wno-poison-system-directories
                -Wno-reserved-id-macro
                -Wno-shadow-uncaptured-local
                -Wno-switch-enum
                -Wno-unused-member-function
                -Wno-used-but-marked-unused
                -Wno-zero-as-null-pointer-constant
        )

        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16)
            target_compile_options(${target} PRIVATE -Wno-unsafe-buffer-usage)
        endif()
    endif()

    if(FAIRSEQ2N_TREAT_WARNINGS_AS_ERRORS)
        target_compile_options(${target}
            PRIVATE
                $<IF:$<COMPILE_LANGUAGE:CUDA>,SHELL:--compiler-options -Werror,-Werror>
        )
    endif()

    if(FAIRSEQ2N_BUILD_FOR_NATIVE)
        target_compile_options(${target} PRIVATE -march=native -mtune=native)
    endif()

    target_compile_options(${target} PRIVATE -fasynchronous-unwind-tables -fstack-protector-strong)

    # Sanitizers do not support source fortification.
    if(NOT FAIRSEQ2N_SANITIZERS OR FAIRSEQ2N_SANITIZERS STREQUAL "nosan")
        target_compile_definitions(${target} PRIVATE $<$<NOT:$<CONFIG:Debug>>:_FORTIFY_SOURCE=2>)
    endif()
endfunction()

function(fairseq2n_set_link_options target)
    cmake_parse_arguments(arg
        #OPTIONS
            "ALLOW_UNDEFINED_SYMBOLS"
        #KEYWORDS
            ""
        #MULTI_VALUE_KEYWORDS
            ""
        #ARGUMENTS
            ${ARGN}
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(${target}
            PRIVATE
                LINKER:--as-needed
                LINKER:--build-id=sha1
                LINKER:-z,noexecstack
                LINKER:-z,now
                LINKER:-z,relro
        )

        if(NOT arg_ALLOW_UNDEFINED_SYMBOLS)
            target_link_options(${target} PRIVATE LINKER:-z,defs)
        endif()

        if(FAIRSEQ2N_TREAT_WARNINGS_AS_ERRORS)
            target_link_options(${target} PRIVATE LINKER:--fatal-warnings)
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        target_link_options(${target} PRIVATE LINKER:-bind_at_load)

        if(NOT arg_ALLOW_UNDEFINED_SYMBOLS)
            target_link_options(${target} PRIVATE LINKER:-undefined,error)
        else()
            target_link_options(${target} PRIVATE
                LINKER:-undefined,dynamic_lookup LINKER:-no_fixup_chains
            )
        endif()

#        if(FAIRSEQ2N_TREAT_WARNINGS_AS_ERRORS)
#            target_link_options(${target} PRIVATE LINKER:-fatal_warnings)
#        endif()
    else()
        message(FATAL_ERROR "fairseq2n supports only Linux and macOS operating systems!")
    endif()

    if(FAIRSEQ2N_PERFORM_LTO)
        set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
endfunction()

function(fairseq2n_set_sanitizers)
    if(NOT FAIRSEQ2N_SANITIZERS OR FAIRSEQ2N_SANITIZERS STREQUAL "nosan")
        return()
    endif()

    foreach(sanitizer IN LISTS FAIRSEQ2N_SANITIZERS)
        if(sanitizer STREQUAL "asan")
            if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                add_compile_definitions(_GLIBCXX_SANITIZE_VECTOR)
            endif()

            list(APPEND sanitizer_flags -fsanitize=address)
        elseif(sanitizer STREQUAL "ubsan")
            list(APPEND sanitizer_flags -fsanitize=undefined)
        elseif(sanitizer STREQUAL "tsan")
            list(APPEND sanitizer_flags -fsanitize=thread)
        else()
            message(FATAL_ERROR "fairseq2n does not support the '${sanitizer}' sanitizer!")
        endif()
    endforeach()

    add_compile_options(${sanitizer_flags} -fno-omit-frame-pointer)

    add_link_options(${sanitizer_flags})
endfunction()
