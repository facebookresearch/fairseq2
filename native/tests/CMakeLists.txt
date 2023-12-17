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
        test_float.cc
        test_memory.cc
        test_span.cc
        data/test_immutable_string.cc
        data/test_tape.cc
        data/detail/test_lru_cache.cc
        utils/test_cast.cc
)

fairseq2n_set_compile_options(tests)

target_link_libraries(tests PRIVATE GTest::gtest_main fairseq2n)

fairseq2n_set_link_options(tests)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # The torch wheel for aarch64 comes with an extra 'torch.libs' directory
    # where the rpath of libopenblas is not correctly set.
    target_link_options(tests PRIVATE LINKER:--allow-shlib-undefined)
endif()

# By default, GTest discovery runs as a post-build step and fails if the targets
# are built with sanitizers enabled.
gtest_discover_tests(tests DISCOVERY_MODE PRE_TEST)
