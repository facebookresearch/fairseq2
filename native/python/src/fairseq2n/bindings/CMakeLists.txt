# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ------------------------------------------------------------
# Target: py_bindings
# ------------------------------------------------------------

Python3_add_library(py_bindings WITH_SOABI)

set_property(TARGET py_bindings PROPERTY OUTPUT_NAME bindings)

target_sources(py_bindings
    PRIVATE
        init.cc
        memory.cc
        data/audio.cc
        data/image.cc
        data/data_pipeline.cc
        data/init.cc
        data/image.cc
        data/string.cc
        data/text/converters.cc
        data/text/init.cc
        data/text/sentencepiece.cc
        data/text/text_reader.cc
        type_casters/data.cc
        type_casters/map_fn.cc
        type_casters/string.cc
        type_casters/torch.cc
)

fairseq2n_set_compile_options(py_bindings)

target_include_directories(py_bindings PRIVATE ${PROJECT_SOURCE_DIR}/python/src)

target_link_libraries(py_bindings
    PRIVATE
        pybind11::module fmt::fmt kuba-zip fairseq2n torch_python
)

fairseq2n_set_link_options(py_bindings ALLOW_UNDEFINED_SYMBOLS)

if(FAIRSEQ2N_PYTHON_DEVEL)
    # We have to use absolute rpath so that the in-source copy of the extension
    # module can find its dependencies under the build tree.
    set_property(TARGET py_bindings PROPERTY BUILD_RPATH_USE_ORIGIN OFF)

    # Copy the extension module to the source tree for
    # `pip install --editable` to work.
    add_custom_target(py_bindings_in_src ALL
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:py_bindings>" "${PROJECT_SOURCE_DIR}/python/src/fairseq2n"
        VERBATIM
    )

    add_dependencies(py_bindings_in_src py_bindings)
endif()

if(FAIRSEQ2N_INSTALL_STANDALONE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(rpath_origin @loader_path)
    else()
        set(rpath_origin \$ORIGIN)
    endif()

    set_property(TARGET py_bindings APPEND PROPERTY INSTALL_RPATH ${rpath_origin}/lib)

    if(FAIRSEQ2N_USE_CUDA)
        # If NVIDIA's CUDA wheels are installed, give them precedence.
        set_property(
            TARGET
                fairseq2n
            APPEND PROPERTY
                INSTALL_RPATH
                    ${rpath_origin}/../../nvidia/cuda_runtime/lib
	)
    endif()
endif()

install(
    TARGETS
        py_bindings
    LIBRARY
        DESTINATION
            .
        COMPONENT
            python
    EXCLUDE_FROM_ALL
)
