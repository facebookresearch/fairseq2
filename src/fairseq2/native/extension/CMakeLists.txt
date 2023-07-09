# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ------------------------------------------------------------
# Target: extension
# ------------------------------------------------------------

Python3_add_library(extension WITH_SOABI)

set_property(TARGET extension PROPERTY OUTPUT_NAME C)

target_sources(extension
    PRIVATE
        memory.cc
        module.cc
        data/audio.cc
        data/data_pipeline.cc
        data/init.cc
        data/string.cc
        data/text/converters.cc
        data/text/init.cc
        data/text/dict_tokenizer.cc
        data/text/sentencepiece.cc
        data/text/text_reader.cc
        type_casters/data.cc
        type_casters/map_fn.cc
        type_casters/string.cc
        type_casters/torch.cc
)

fairseq2_set_compile_options(extension)

target_include_directories(extension PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(extension
    PRIVATE
        pybind11::module fairseq2 fmt::fmt torch_python kuba-zip
)

fairseq2_set_link_options(extension ALLOW_UNDEFINED_SYMBOLS)

if(FAIRSEQ2_PYTHON_DEVEL)
    # We have to use absolute rpath so that the in-source copy of the extension
    # module can find its dependencies under the build tree.
    set_property(TARGET extension PROPERTY BUILD_RPATH_USE_ORIGIN OFF)

    # Copy the extension module to the source tree for
    # `pip install --editable` to work.
    add_custom_target(extension_in_source ALL
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:extension>" "${PROJECT_SOURCE_DIR}/src/fairseq2"
        VERBATIM
    )

    add_dependencies(extension_in_source extension)
endif()

if(FAIRSEQ2_BUILD_FOR_WHEEL_BUNDLE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(rpath_origin @loader_path)
    else()
        set(rpath_origin \$ORIGIN)
    endif()

    set_property(
        TARGET
            extension
        APPEND PROPERTY
            INSTALL_RPATH
                ${rpath_origin}/lib
                ${rpath_origin}/../../..
                ${rpath_origin}/../torch/lib
    )
endif()

install(
    TARGETS
        extension
    LIBRARY
        DESTINATION
            fairseq2
        COMPONENT
            python
    EXCLUDE_FROM_ALL
)
