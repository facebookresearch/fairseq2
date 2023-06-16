// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <fairseq2/native/data/text/detail/string_utils.h>
#include <fairseq2/native/data/immutable_string.h>

#include <gtest/gtest.h>

#include <ATen/Tensor.h>
#include <ATen/Functions.h>

#include <stdexcept>

using namespace fairseq2;
using namespace fairseq2::detail;

TEST(test_string_utils, parse_simple_string)
{
    auto actual = parse_tensor(immutable_string{"8 9 10 11 12 13 14"});
    at::Tensor t = at::arange(8, 15);
    EXPECT_TRUE(t.equal(actual));
}

TEST(test_string_utils, parse_negative_digits_string)
{
    auto actual = parse_tensor(immutable_string{"13 -245 78 0 -12"});
    auto expected = at::tensor({13, -245, 78, 0, -12}, {at::kLong});
    EXPECT_TRUE(actual.equal(expected));
}

TEST(test_string_utils, empty_string)
{
    auto actual = parse_tensor(immutable_string{" "});
    at::Tensor t = at::arange(0);
    EXPECT_TRUE(t.equal(actual));
}

TEST(test_string_utils, throws_on_bad_content)
{
    EXPECT_THROW(parse_tensor(immutable_string{"1 3 test 5"}), std::invalid_argument);
}
