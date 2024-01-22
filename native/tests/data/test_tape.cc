// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <fairseq2n/data/tape.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <fairseq2n/float.h>
#include <fairseq2n/data/immutable_string.h>

using namespace fairseq2n;

TEST(test_tape, record_and_read_work)
{
    tape t{};

    bool a = true;
    std::int32_t b = 4;
    std::int64_t c = 8;
    float32 d = 0.1F;
    float64 e = 3.2;
    immutable_string f = "hello";
    data_list g{data{"a"}, data{"b"}};
    data_dict h{{"a", data{"a"}}, {"b", data{"b"}}};

    t.record(a);
    t.record(b);
    t.record(c);
    t.record(d);
    t.record(e);
    t.record(f);
    t.record(g);
    t.record(h);

    for (std::size_t i = 0; i < 3; i++) {
        t.rewind();

        EXPECT_EQ(a, t.read<bool>());
        EXPECT_EQ(b, t.read<std::int32_t>());
        EXPECT_EQ(c, t.read<std::int64_t>());
        EXPECT_EQ(d, t.read<float32>());
        EXPECT_EQ(e, t.read<float64>());
        EXPECT_EQ(f, t.read<immutable_string>());

        auto x = t.read<data_list>();

        EXPECT_EQ(x.size(), 2);

        EXPECT_TRUE(x[0].is_string());
        EXPECT_TRUE(x[1].is_string());

        EXPECT_EQ(x[0].as_string(), g[0].as_string());
        EXPECT_EQ(x[1].as_string(), g[1].as_string());

        auto y = t.read<data_dict>();

        EXPECT_EQ(y.size(), 2);

        EXPECT_TRUE(y["a"].is_string());
        EXPECT_TRUE(y["b"].is_string());

        EXPECT_EQ(y["a"].as_string(), h["a"].as_string());
        EXPECT_EQ(y["b"].as_string(), h["b"].as_string());
    }
}

TEST(test_tape, record_and_read_work_when_inputs_are_composite)
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

TEST(test_tape, read_throws_exception_when_record_is_not_called)
{
    tape t{};

    EXPECT_THROW(t.read_data(), std::invalid_argument);
}

TEST(test_tape, read_throws_exception_when_record_is_of_different_type)
{
    tape t{};

    t.record("hello");

    t.rewind();

    EXPECT_THROW(t.read<std::int32_t>(), std::invalid_argument);
}

TEST(test_tape, read_throws_exception_when_end_of_tape_is_reached)
{
    tape t{};

    t.record(4);

    // No rewind.
    EXPECT_THROW(t.read<std::int32_t>(), std::invalid_argument);
}
