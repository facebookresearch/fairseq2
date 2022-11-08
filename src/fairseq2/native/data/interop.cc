// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/interop.h"

#include <algorithm>

#include "fairseq2/native/utils/text.h"

using fairseq2::detail::memory_block;
using fairseq2::detail::mutable_memory_block;

namespace fairseq2 {
namespace {

constexpr char null_char = '\0';

memory_block
copy_string(std::string_view s)
{
    span src = s;

    mutable_memory_block block = detail::allocate_host_memory(src.size_bytes() + sizeof(char));

    auto dst = block.cast<char>();

    auto last_pos = std::copy(src.begin(), src.end(), dst.begin());

    *last_pos = null_char;

    return block;
}

}  // namespace

// Constructor
istring::istring(std::string_view s)
    : bits_{copy_string(s)}
{}

istring::const_pointer
istring::c_str() const noexcept
{
    if (empty())
        return &null_char;

    return data();
}

std::size_t
istring::get_code_point_length() const
{
    return detail::get_code_point_length(view());
}

}  // namespace fairseq2
