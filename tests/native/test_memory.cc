// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <fairseq2/native/memory.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

using namespace fairseq2;

TEST(test_memory_block, default_constructed_is_empty)
{
    memory_block b{};

    EXPECT_EQ(b.data(), nullptr);
    EXPECT_EQ(b.size(), 0);

    EXPECT_TRUE(b.empty());
}

TEST(test_memory_block, can_construct_from_data_and_size)
{
    std::array<std::byte, 5> a{};

    memory_block b{a.data(), a.size()};

    EXPECT_EQ(b.data(), a.data());
    EXPECT_EQ(b.size(), a.size());

    EXPECT_FALSE(b.empty());
}

static void
test_dealloc(const void *ptr, std::size_t size) noexcept
{
    // Used by the next test to check the deallocator call.
    *static_cast<std::size_t *>(const_cast<void *>(ptr)) = size;  // NOLINT
}

TEST(test_memory_block, destructor_calls_deallocator)
{
    std::array<std::byte, sizeof(std::size_t)> a{};

    auto *data = reinterpret_cast<std::size_t *>(a.data());

    EXPECT_EQ(*data, 0);

    {
        memory_block b{a.data(), a.size(), test_dealloc};
    }

    EXPECT_EQ(*data, a.size());
}

TEST(test_memory_block, cast_returns_correct_range)
{
    std::array<std::byte, 16> a{};

    memory_block b{a.data(), a.size()};

    span s = b.cast<const std::int32_t>();

    EXPECT_EQ(static_cast<const void *>(s.data()), static_cast<const void *>(a.data()));

    EXPECT_EQ(s.size(), a.size() / sizeof(std::int32_t));

    EXPECT_EQ(s.size_bytes(), a.size());
}
