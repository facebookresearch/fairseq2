// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <fairseq2/native/span.h>

#include <array>
#include <cstdint>

#include <gtest/gtest.h>

using fairseq2::span;

TEST(test_span, default_constructed_is_empty)
{
    span<const char> s{};

    EXPECT_EQ(s.data(), nullptr);
    EXPECT_EQ(s.size(), 0);

    EXPECT_TRUE(s.empty());
}

TEST(test_span, can_construct_from_iterator_pair)
{
    std::array<const char, 5> a = {'a', 'b', 'c', 'd', 'e'};

    span s{a.begin(), a.end()};

    EXPECT_EQ(s.data(), a.data());
    EXPECT_EQ(s.size(), a.size());

    EXPECT_FALSE(s.empty());
}

TEST(test_span, can_construct_from_data_and_size)
{
    std::array<const char, 5> a = {'a', 'b', 'c', 'd', 'e'};

    span s{a.data(), a.size()};

    EXPECT_EQ(s.data(), a.data());
    EXPECT_EQ(s.size(), a.size());

    EXPECT_FALSE(s.empty());
}

TEST(test_span, can_construct_from_container)
{
    std::array<const char, 5> a = {'a', 'b', 'c', 'd', 'e'};

    span s{a};

    EXPECT_EQ(s.data(), a.data());
    EXPECT_EQ(s.size(), a.size());

    EXPECT_FALSE(s.empty());
}

TEST(test_span, subspan_returns_correct_range)
{
    std::array<const char, 5> a = {'a', 'b', 'c', 'd', 'e'};

    span s{a};

    s = s.subspan(2);

    EXPECT_EQ(s.data(), a.data() + 2);
    EXPECT_EQ(s.size(), a.size() - 2);

    EXPECT_FALSE(s.empty());
}

TEST(test_span, subspan_with_count_returns_correct_range)
{
    std::array<const char, 5> a = {'a', 'b', 'c', 'd', 'e'};

    span s{a};

    s = s.subspan(1, 2);

    EXPECT_EQ(s.data(), a.data() + 1);
    EXPECT_EQ(s.size(), 2);

    EXPECT_FALSE(s.empty());
}

TEST(test_span, first_returns_correct_range)
{
    std::array<const char, 5> a = {'a', 'b', 'c', 'd', 'e'};

    span s{a};

    s = s.first(2);

    EXPECT_EQ(s.data(), a.data());
    EXPECT_EQ(s.size(), 2);

    EXPECT_FALSE(s.empty());
}

TEST(test_span, last_returns_correct_range)
{
    std::array<const char, 5> a = {'a', 'b', 'c', 'd', 'e'};

    span s{a};

    s = s.last(2);

    EXPECT_EQ(s.data(), a.data() + a.size() - 2);
    EXPECT_EQ(s.size(), 2);

    EXPECT_FALSE(s.empty());
}

TEST(test_span, size_bytes_returns_correct_value)
{
    std::array<const std::int32_t, 5> a = {1, 2, 3, 4, 5};

    span s{a};

    EXPECT_EQ(s.size_bytes(), a.size() * sizeof(std::int32_t));
}
