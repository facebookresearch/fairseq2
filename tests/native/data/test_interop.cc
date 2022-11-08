// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <fairseq2/native/data/interop.h>
#include <fairseq2/native/utils/text.h>

#include <gtest/gtest.h>

using fairseq2::invalid_utf8;
using fairseq2::istring;

//
// The remaining API of `istring` is tested in Python.
//
TEST(test_istring, construct_throws_exception_if_string_is_invalid_utf8)
{
    istring s{"\xfe\xfe\xff\xff"};

    EXPECT_THROW(s.get_code_point_length(), invalid_utf8);
}
