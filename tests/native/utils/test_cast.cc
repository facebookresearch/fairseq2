// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <fairseq2/native/utils/cast.h>

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include <fairseq2/native/float.h>

using namespace fairseq2;
using namespace fairseq2::detail;

TEST(test_cast, try_narrow_returns_true_if_value_within_range)
{
    std::int64_t a = 100;
    std::int32_t b = 0;

    EXPECT_TRUE(try_narrow(a, b));

    EXPECT_EQ(b, 100);

    float64 c = 12.0;
    float32 d = 0;

    EXPECT_TRUE(try_narrow(c, d));

    EXPECT_EQ(d, 12.0);
}

TEST(test_cast, try_narrow_returns_false_if_value_outside_of_range)
{
    std::int64_t a = std::numeric_limits<std::int64_t>::max();
    std::int32_t b = 0;

    EXPECT_FALSE(try_narrow(a, b));

    float64 c = std::numeric_limits<float64>::max();
    float32 d = 0;

    EXPECT_FALSE(try_narrow(c, d));
}
