// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <fairseq2/native/data/immutable_string.h>

#include <functional>
#include <memory>

#include <gtest/gtest.h>

#include <fairseq2/native/data/text/detail/utf.h>

using namespace fairseq2;

// immutable_string (also see Python tests)
TEST(test_immutable_string, constructor_throws_exception_if_string_is_invalid_utf8)
{
    immutable_string s{"\xfe\xfe\xff\xff"};

    EXPECT_THROW(s.get_code_point_length(), invalid_utf8_error);
}

TEST(test_immutable_string, copy_constructor_shares_memory)
{
    immutable_string s1 = "foo";

    immutable_string s2 = s1;  // NOLINT(performance-unnecessary-copy-initialization)

    EXPECT_EQ(s1.data(), s2.data());
}
