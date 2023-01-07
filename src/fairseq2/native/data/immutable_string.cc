// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/immutable_string.h"

#include <algorithm>

#include "fairseq2/native/data/text/detail/utf.h"

using namespace fairseq2::detail;

namespace fairseq2 {

immutable_string::immutable_string(std::string_view s)
    : storage_{copy_string(s)}
{}

std::size_t
immutable_string::get_code_point_length() const
{
    return compute_code_point_length(view());
}

memory_block
immutable_string::copy_string(std::string_view s)
{
    writable_memory_block blk = allocate_memory(s.size());

    std::copy(s.begin(), s.end(), blk.cast<value_type>().begin());

    return blk;
}

}  // namespace fairseq2
