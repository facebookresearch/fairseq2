// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <stdexcept>
#include <cctype>

#include "fairseq2/native/data/immutable_string.h"
#include "fairseq2/native/data/text/detail/utf.h"
#include <fairseq2/native/utils/cast.h>

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

std::vector<immutable_string>
immutable_string::split(char separator) const
{
    std::vector<immutable_string> result;
    std::string_view s = view();
    std::size_t offset = 0;

    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == separator) {
            if (offset != i)
                result.emplace_back(storage_.share_slice(offset, i - offset));

            offset = i + 1;
        }
    }

    if (offset != s.size())
        result.push_back(remove_prefix(offset));

    return result;
}

std::int32_t
immutable_string::to_int32() const
{
    std::string_view s = view();
    if (s.empty())
        throw std::runtime_error("Trying to cast empty string to int32");

    std::int32_t sign = 1;
    std::ptrdiff_t offset = 0;
    if (s[0] == '-') {
        sign = -1;
        offset = 1;
    }

    std::int32_t result = 0;
    std::int32_t decimal_factor = 1;
    for (std::ptrdiff_t i = ssize(s) - 1; i >= offset; --i) {
        auto index = static_cast<std::size_t>(i);
        if (std::isdigit(s[index]) == 0)
            throw std::runtime_error("Unexcpeted non digit character in string.");

        result += decimal_factor * (s[index] - '0');
        decimal_factor *= 10;
    }

    return sign * result;
}

}  // namespace fairseq2
