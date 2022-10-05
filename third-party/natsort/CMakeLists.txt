# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_library(natsort OBJECT strnatcmp.c)

set_target_properties(natsort PROPERTIES
    C_VISIBILITY_PRESET
        hidden
    POSITION_INDEPENDENT_CODE
        ON
)

target_include_directories(natsort SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
