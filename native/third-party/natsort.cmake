# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

macro(fairseq2n_add_natsort)
    add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/natsort EXCLUDE_FROM_ALL)
endmacro()
