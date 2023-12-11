// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/immutable_string.h"

#include <algorithm>

#include "fairseq2n/data/text/detail/utf.h"
#include "fairseq2n/detail/exception.h"

using namespace fairseq2n::detail;

namespace fairseq2n {

immutable_string::immutable_string(std::string_view s)
  : storage_{copy_string(s)}
{}

std::size_t
immutable_string::get_code_point_length() const
{
    try {
        return compute_code_point_length(view());
    } catch (const std::invalid_argument &) {
        throw_<invalid_utf8_error>("The string has one or more invalid UTF-8 code points.");
    }
}

memory_block
immutable_string::copy_string(std::string_view s)
{
    writable_memory_block block = allocate_memory(s.size());

    std::copy(s.begin(), s.end(), block.cast<value_type>().begin());

    return block;
}

std::vector<immutable_string>
immutable_string::split(char separator) const
{
    std::vector<immutable_string> output{};

    split(separator, [&output](immutable_string &&s)
    {
        output.push_back(std::move(s));

        return true;
    });

    return output;
}

void
immutable_string::split(
    char separator, const std::function<bool(immutable_string &&)> &handler) const
{
    std::string_view s = view();

    std::size_t offset = 0;

    for (std::size_t char_idx = 0; char_idx < s.size(); ++char_idx) {
        if (s[char_idx] == separator) {
            immutable_string part{storage_.share_slice(offset, char_idx - offset)};

            if (!handler(std::move(part)))
                return;

            offset = char_idx + 1;
        }
    }

    handler(remove_prefix(offset));
}

invalid_utf8_error::~invalid_utf8_error() = default;

}  // namespace fairseq2n
