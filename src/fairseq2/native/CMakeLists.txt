# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ------------------------------------------------------------
# Target: fairseq2
# ------------------------------------------------------------

fairseq2_add_target(fairseq2 SHARED_LIBRARY)

target_sources(fairseq2
    PRIVATE
        foo.cc
)

add_library(fairseq2::fairseq2 ALIAS fairseq2)
