// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <fairseq2n/data/detail/lru_cache.h>

#include <cstdint>

#include <gtest/gtest.h>

#include <fairseq2n/data/immutable_string.h>

using namespace fairseq2n::detail;

TEST(test_lru_cache, add_and_maybe_get_work)
{
    lru_cache<std::int32_t> cache{/*capacity=*/5};

    cache.add("1", 1);
    cache.add("2", 2);
    cache.add("3", 3);
    cache.add("4", 4);
    cache.add("5", 5);

    // Mark 1 as the least-recently-used entry.
    EXPECT_NE(cache.maybe_get("1"), nullptr);

    EXPECT_EQ(cache.size(), 5);

    // Exceed capacity; we expect 2 to be evicted.
    cache.add("6", 6);

    EXPECT_EQ(cache.size(), 5);

    EXPECT_EQ(cache.maybe_get("2"), nullptr);

    // Exceed capacity again; we expect 3 to be evicted.
    cache.add("7", 7);

    EXPECT_EQ(cache.size(), 5);

    EXPECT_EQ(cache.maybe_get("3"), nullptr);

    // This time touch 4, and mark it as least-recently-used.
    EXPECT_NE(cache.maybe_get("4"), nullptr);

    // Exceed capacity again; we expect 5 to be evicted.
    cache.add("8", 8);

    EXPECT_EQ(cache.size(), 5);

    EXPECT_EQ(cache.maybe_get("5"), nullptr);

    cache.clear();

    EXPECT_EQ(cache.size(), 0);
}
