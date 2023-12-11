# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_subdirectory(bindings)

# ------------------------------------------------------------
# Library Configuration
# ------------------------------------------------------------

if(FAIRSEQ2N_SUPPORT_IMAGE)
    set(SUPPORTS_IMAGE "True")
else()
    set(SUPPORTS_IMAGE "False")
endif()

if(FAIRSEQ2N_USE_CUDA)
    set(USES_CUDA "True")

    set(CUDA_VERSION "(${CUDAToolkit_VERSION_MAJOR}, ${CUDAToolkit_VERSION_MINOR})")
else()
    set(USES_CUDA "False")

    set(CUDA_VERSION "None")
endif()

configure_file(config.py.in ${CMAKE_CURRENT_SOURCE_DIR}/config.py @ONLY)
