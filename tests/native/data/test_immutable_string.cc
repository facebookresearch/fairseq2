// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <fairseq2/native/data/immutable_string.h>

#include <functional>
#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>

#include <fairseq2/native/data/text/detail/utf.h>

using namespace fairseq2;

// Also see the Python tests.
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

TEST(test_imutable_string, to_int32_postive_integers)
{
    immutable_string s1 = "123";
    EXPECT_EQ(s1.to_int32(), 123);

    immutable_string s2 = "3195";
    EXPECT_EQ(s2.to_int32(), 3195);

    immutable_string s3 = "0";
    EXPECT_EQ(s3.to_int32(), 0);

    immutable_string s4 = "7";
    EXPECT_EQ(s4.to_int32(), 7);

    immutable_string s5 = "193747";
    EXPECT_EQ(s5.to_int32(), 193747);
}

TEST(test_imutable_string, to_int32_negative_integers)
{
    immutable_string s1 = "-123";
    EXPECT_EQ(s1.to_int32(), -123);

    immutable_string s2 = "-3195";
    EXPECT_EQ(s2.to_int32(), -3195);

    immutable_string s3 = "-0";
    EXPECT_EQ(s3.to_int32(), 0);

    immutable_string s4 = "-7";
    EXPECT_EQ(s4.to_int32(), -7);

    immutable_string s5 = "-193747";
    EXPECT_EQ(s5.to_int32(), -193747);
}

TEST(test_imutable_string, to_int32_throws_at_bad_format)
{
    immutable_string s1 = "123x4";
    EXPECT_THROW(s1.to_int32(), std::runtime_error);

    immutable_string s2 = " ";
    EXPECT_THROW(s2.to_int32(), std::runtime_error);

    immutable_string s3 = "a1234";
    EXPECT_THROW(s3.to_int32(), std::runtime_error);
}
