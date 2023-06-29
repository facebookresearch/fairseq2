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
    writable_memory_block b = allocate_memory(s.size());

    std::copy(s.begin(), s.end(), b.cast<value_type>().begin());

    return b;
}

std::vector<immutable_string>
immutable_string::split(char separator) const
{
    std::vector<immutable_string> r{};

    split(separator, [&r](immutable_string &&s) {
        r.push_back(std::move(s));
    });

    return r;
}

void
immutable_string::split(
    char separator, const std::function<void(immutable_string &&)> &handler) const
{
    std::string_view s = view();

    std::size_t offset = 0;

    for (std::size_t i = 0; i < s.size(); i++) {
        if (s[i] == separator) {
            if (offset != i) {
                immutable_string part{storage_.share_slice(offset, i - offset)};

                handler(std::move(part));
            }

            offset = i + 1;
        }
    }

    if (offset != s.size())
        handler(remove_prefix(offset));
}

}  // namespace fairseq2
