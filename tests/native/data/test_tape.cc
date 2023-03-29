// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <fairseq2/native/data/tape.h>

#include <cstdint>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include <fairseq2/native/float.h>
#include <fairseq2/native/data/immutable_string.h>

using namespace fairseq2;

TEST(test_tape, records_and_reads_primitives_as_expected)
{
    tape t{};

    bool a = true;
    std::int32_t b = 4;
    std::int64_t c = 8;
    float32 d = 0.1F;
    float64 e = 3.2;
    immutable_string f = "hello";

    t.record(a);
    t.record(b);
    t.record(c);
    t.record(d);
    t.record(e);
    t.record(f);

    for (std::size_t i = 0; i < 3; i++) {
        t.rewind();

        EXPECT_EQ(a, t.read<bool>());
        EXPECT_EQ(b, t.read<std::int32_t>());
        EXPECT_EQ(c, t.read<std::int64_t>());
        EXPECT_EQ(d, t.read<float32>());
        EXPECT_EQ(e, t.read<float64>());
        EXPECT_EQ(f, t.read<immutable_string>());
    }
}

TEST(test_tape, records_and_reads_composite_objects_as_expected)
{
    using T = std::vector<std::int32_t>;

    T a{1, 2, 3};

    std::optional<T> b = std::move(a);

    tape t{};

    t.record(b);

    for (std::size_t i = 0; i < 3; i++) {
        t.rewind();

        auto c = t.read<std::optional<T>>();

        ASSERT_TRUE(c);

        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        EXPECT_EQ(*b, *c);
    }
}

TEST(test_tape, read_without_record_raises_error)
{
    tape t{};

    EXPECT_THROW(t.read_data(), corrupt_tape_error);
}

TEST(test_tape, unpaired_read_and_record_raises_error)
{
    tape t{};

    t.record("hello");

    t.rewind();

    EXPECT_THROW(t.read<std::int32_t>(), corrupt_tape_error);
}

TEST(test_tape, reading_end_of_tape_raises_error)
{
    tape t{};

    t.record(4);

    // No rewind.
    EXPECT_THROW(t.read<std::int32_t>(), corrupt_tape_error);
}
