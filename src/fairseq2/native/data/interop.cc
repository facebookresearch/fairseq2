// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/interop.h"

#include <algorithm>

#include "fairseq2/native/error.h"
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

std::size_t
std::hash<fairseq2::ivariant>::operator()(const fairseq2::ivariant &value) const
{
    if (value.is_uninitialized())
        return get_hash(&value);

    if (value.is_none())
        return 0;

    if (value.is_bool())
        return get_hash(value.as_bool());

    if (value.is_int())
        return get_hash(value.as_int());

    if (value.is_double())
        return get_hash(value.as_double());

    if (value.is_string())
        return get_hash(value.as_string());

    if (value.is_tensor())
        return get_hash(value.as_tensor().unsafeGetTensorImpl());

    fairseq2::detail::unreachable();
}
