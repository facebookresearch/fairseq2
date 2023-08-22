// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <fairseq2n/memory.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

using namespace fairseq2n;

TEST(test_memory_block, constructor_works)
{
    memory_block b{};

    EXPECT_EQ(b.data(), nullptr);
    EXPECT_EQ(b.size(), 0);

    EXPECT_TRUE(b.empty());
}

TEST(test_memory_block, constructor_works_when_data_and_size_are_specified)
{
    std::array<std::byte, 5> a{};

    memory_block b{a.data(), a.size()};

    EXPECT_EQ(b.data(), a.data());
    EXPECT_EQ(b.size(), a.size());

    EXPECT_FALSE(b.empty());
}

TEST(test_memory_block, move_constructor_works)
{
    std::array<std::byte, 5> a{};

    memory_block b{a.data(), a.size()};

    EXPECT_EQ(b.data(), a.data());
    EXPECT_EQ(b.size(), a.size());

    memory_block c = std::move(b);

    EXPECT_EQ(c.data(), a.data());
    EXPECT_EQ(c.size(), a.size());

    // b has been moved. Normally it's not valid to access it, but we want to check
    // that even if we do we have a 0 size string and don't segfault.
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move)
    EXPECT_EQ(b.data(), nullptr);
    EXPECT_EQ(b.size(), 0);

    EXPECT_TRUE(b.empty());
}

TEST(test_memory_block, move_operator_works)
{
    std::array<std::byte, 5> a{};

    memory_block b{a.data(), a.size()};

    EXPECT_EQ(b.data(), a.data());
    EXPECT_EQ(b.size(), a.size());

    memory_block c{};

    c = std::move(b);

    EXPECT_EQ(c.data(), a.data());
    EXPECT_EQ(c.size(), a.size());

    // b has been moved. Normally it's not valid to access it, but we want to check
    // that even if we do we have a 0 size string and don't segfault.
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move)
    EXPECT_EQ(b.data(), nullptr);
    EXPECT_EQ(b.size(), 0);

    EXPECT_TRUE(b.empty());
}

static void
test_dealloc(const void *ptr, std::size_t size, void *) noexcept
{
    // Used by the next test to check the deallocator call.
    *static_cast<std::size_t *>(const_cast<void *>(ptr)) = size;  // NOLINT
}

TEST(test_memory_block, destructor_works)
{
    std::array<std::byte, sizeof(std::size_t)> a{};

    auto *data = reinterpret_cast<std::size_t *>(a.data());

    EXPECT_EQ(*data, 0);

    {
        memory_block b{a.data(), a.size(), nullptr, test_dealloc};
    }

    EXPECT_EQ(*data, a.size());
}

TEST(test_memory_block, cast_works)
{
    std::array<std::byte, 16> a{};

    memory_block b{a.data(), a.size()};

    span s = b.cast<const std::int32_t>();

    EXPECT_EQ(static_cast<const void *>(s.data()), static_cast<const void *>(a.data()));

    EXPECT_EQ(s.size(), a.size() / sizeof(std::int32_t));

    EXPECT_EQ(s.size_bytes(), a.size());
}
