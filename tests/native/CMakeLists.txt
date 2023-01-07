# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ------------------------------------------------------------
# Target: tests
# ------------------------------------------------------------

add_executable(tests)

set_property(TARGET tests PROPERTY OUTPUT_NAME run-tests)

target_sources(tests
    PRIVATE
        test_memory.cc
        test_span.cc
        data/test_immutable_string.cc
)

fairseq2_set_compile_options(tests)

target_link_libraries(tests PRIVATE GTest::gtest_main fairseq2)

fairseq2_set_link_options(tests)

gtest_discover_tests(tests)
