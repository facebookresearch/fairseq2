// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <fairseq2n/float.h>

#include <gtest/gtest.h>

using namespace fairseq2n;

TEST(test_cast, are_close_works_when_inputs_are_equal)
{
    float32 a = 1.0F;
    float32 b = 1.0F;

    EXPECT_TRUE(are_close(a, b));

    float64 c = 1.0;
    float64 d = 1.0;

    EXPECT_TRUE(are_close(c, d));
}

TEST(test_cast, are_close_works_when_inputs_are_within_relative_distance)
{
    float32 a = 3.0F;
    // This is the maximum tolerance we have for the relative difference
    // between the two numbers.
    float32 b = 3.0F + (a * 0.0001F);

    EXPECT_TRUE(are_close(a, b));

    // This number should be treated as equal.
    float32 c = 3.0F + (a * 0.000001F);

    EXPECT_TRUE(are_close(a, c));
}

TEST(test_cast, are_close_works_when_inputs_are_outside_of_relative_distance)
{
    float32 a = 3.0F;
    // This is beyond our tolerance threshold.
    float32 b = 3.0F + (a * 0.00011F);

    EXPECT_FALSE(are_close(a, b));
}
