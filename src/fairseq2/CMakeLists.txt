# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_subdirectory(native)

# ------------------------------------------------------------
# Module: config
# ------------------------------------------------------------

fairseq2_add_target(py_config PYTHON_MODULE OUTPUT_NAME config)

target_sources(py_config PRIVATE config.cc)

target_link_libraries(py_config PRIVATE pybind11::module fairseq2)
