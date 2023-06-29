// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/processors/string_to_integer_converter.h"

#include <charconv>
#include <stdexcept>

#include <fmt/core.h>

#include "fairseq2/native/data/immutable_string.h"

namespace fairseq2 {

data
string_to_integer_converter::process(data &&d) const
{
    if (!d.is_string())
        throw std::invalid_argument{"The input data must be of type string."};

    immutable_string s = d.as_string();

    const char *end = s.data() + s.size();

    std::int64_t parsed_value{};

    std::from_chars_result r = std::from_chars(s.data(), end, parsed_value, base_);
    if (r.ec == std::errc{} && r.ptr == end)
        return parsed_value;

    if (r.ec == std::errc::result_out_of_range)
        throw std::invalid_argument{
            fmt::format("The input string must be a signed 64-bit integer, but is '{}' instead.", s)};
    else
        throw std::invalid_argument{
            fmt::format("The input string must be an integer, but is '{}' instead.", s)};
}

}  // namespace fairseq2
