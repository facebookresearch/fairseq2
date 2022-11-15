// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <fairseq2/native/data/interop.h>

#include <functional>
#include <memory>

#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <fairseq2/native/utils/text.h>

using fairseq2::idict;
using fairseq2::ilist;
using fairseq2::invalid_utf8;
using fairseq2::istring;
using fairseq2::ivariant;

// istring (also see Python tests)
TEST(test_istring, construct_throws_exception_if_string_is_invalid_utf8)
{
    istring s{"\xfe\xfe\xff\xff"};

    EXPECT_THROW(s.get_code_point_length(), invalid_utf8);
}

// ivariant
TEST(test_ivariant, default_constructed_is_uninitialized)
{
    ivariant v{};

    EXPECT_TRUE(v.is_uninitialized());
}

TEST(test_ivariant, string_constructed_shares_memory_with_argument)
{
    istring s = "foo";

    ivariant v = s;

    EXPECT_TRUE(v.is_string());

    EXPECT_EQ(v.as_string().data(), s.data());
}

TEST(test_ivariant, eq_returns_true_if_values_are_equal)
{
    std::hash<ivariant> h{};

    ivariant v1 = ivariant::none();
    ivariant v2 = ivariant::none();

    EXPECT_EQ(v1, v2);

    EXPECT_EQ(h(v1), h(v2));

    bool b = false;

    v1 = b;
    v2 = b;

    EXPECT_EQ(v1, v2);

    EXPECT_EQ(h(v1), h(v2));

    std::int64_t i = 5;

    v1 = i;
    v2 = i;

    EXPECT_EQ(v1, v2);

    EXPECT_EQ(h(v1), h(v2));

    double f = 3.0;

    v1 = f;
    v2 = f;

    EXPECT_EQ(v1, v2);

    EXPECT_EQ(h(v1), h(v2));

    istring s1 = "foo";
    istring s2 = "foo";

    v1 = s1;
    v2 = s2;

    EXPECT_EQ(v1, v2);

    EXPECT_EQ(h(v1), h(v2));

    at::Tensor t = at::ones({10, 10});

    v1 = t;
    v2 = t;

    EXPECT_EQ(v1, v2);

    EXPECT_EQ(h(v1), h(v2));

    ilist l1{"a", "b", "c"};
    ilist l2{"a", "b", "c"};

    v1 = l1;
    v2 = l2;

    EXPECT_EQ(v1, v2);

    EXPECT_NE(h(v1), h(v2));

    auto l = std::make_shared<ilist>(l1);

    v1 = l;
    v2 = l;

    EXPECT_EQ(v1, v2);

    EXPECT_EQ(h(v1), h(v2));

    idict d1{{"a", 1.0}};
    idict d2{{"a", 1.0}};

    v1 = d1;
    v2 = d2;

    EXPECT_EQ(v1, v2);

    EXPECT_NE(h(v1), h(v2));

    auto d = std::make_shared<idict>(d1);

    v1 = d;
    v2 = d;

    EXPECT_EQ(v1, v2);

    EXPECT_EQ(h(v1), h(v2));
}

TEST(test_ivariant, eq_returns_false_if_values_are_not_equal)
{
    std::hash<ivariant> h{};

    ivariant v1{};
    ivariant v2{};

    EXPECT_NE(v1, v2);  // Uninitialized variables are never equal.

    EXPECT_NE(h(v1), h(v2));

    bool b1 = false;
    bool b2 = true;

    v1 = b1;
    v2 = b2;

    EXPECT_NE(v1, v2);

    EXPECT_NE(h(v1), h(v2));

    std::int64_t i1 = 5;
    std::int64_t i2 = 6;

    v1 = i1;
    v2 = i2;

    EXPECT_NE(v1, v2);

    EXPECT_NE(h(v1), h(v2));

    double f1 = 3.0;
    double f2 = 4.0;

    v1 = f1;
    v2 = f2;

    EXPECT_NE(v1, v2);

    EXPECT_NE(h(v1), h(v2));

    istring s1 = "foo";
    istring s2 = "fo0";

    v1 = s1;
    v2 = s2;

    EXPECT_NE(v1, v2);

    EXPECT_NE(h(v1), h(v2));

    at::Tensor t1 = at::ones({10, 10});
    at::Tensor t2 = at::ones({10, 10});

    v1 = t1;
    v2 = t2;

    EXPECT_NE(v1, v2);

    EXPECT_NE(h(v1), h(v2));

    ilist l1{"a", "b", "c"};
    ilist l2{"d", "e", "f"};

    v1 = l1;
    v2 = l2;

    EXPECT_NE(v1, v2);

    EXPECT_NE(h(v1), h(v2));

    idict d1{{"a", 1.0}};
    idict d2{{"b", 1.0}};

    v1 = d1;
    v2 = d2;

    EXPECT_NE(v1, v2);

    EXPECT_NE(h(v1), h(v2));
}

TEST(test_ivariant, eq_returns_false_if_types_are_different)
{
    std::hash<ivariant> h{};

    std::int64_t i = 1;

    double f = 3.0;

    at::Tensor t = at::ones({10, 10});

    ilist l{"a", "b", "c"};

    idict d{{"a", 1.0}};

    ivariant v1{};
    ivariant v2 = i;

    EXPECT_NE(v1, v2);

    EXPECT_NE(h(v1), h(v2));

    v1 = f;
    v2 = i;

    EXPECT_NE(v1, v2);

    EXPECT_NE(h(v1), h(v2));

    v1 = t;
    v2 = f;

    EXPECT_NE(v1, v2);

    EXPECT_NE(h(v1), h(v2));

    v1 = ivariant::none();
    v2 = i;

    EXPECT_NE(v1, v2);

    EXPECT_NE(h(v1), h(v2));

    v1 = t;
    v2 = l;

    EXPECT_NE(v1, v2);

    EXPECT_NE(h(v1), h(v2));

    v1 = l;
    v2 = d;

    EXPECT_NE(v1, v2);

    EXPECT_NE(h(v1), h(v2));
}
