# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

fairseq2_add_target(tests TEST_EXECUTABLE)

target_sources(tests
    PRIVATE
        native/utils/test_foo.cc
)

target_link_libraries(tests PRIVATE Catch2::Catch2WithMain fairseq2)

# Register the Catch2 tests with CTest.
catch_discover_tests(tests)
