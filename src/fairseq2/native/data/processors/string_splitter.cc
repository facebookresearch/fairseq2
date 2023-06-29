// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/processors/string_splitter.h"

#include <stdexcept>
#include <utility>

#include <fmt/core.h>

#include "fairseq2/native/data/data.h"

namespace fairseq2 {

string_splitter::string_splitter(char sep, std::optional<std::vector<std::string>> names) noexcept
  : sep_{sep}
{
    if (names)
        names_ = *std::move(names);
}

data
string_splitter::process(data &&d) const
{
    if (!d.is_string())
        throw std::invalid_argument{"The input data must be of type string."};

    std::vector<data> fields{};

    d.as_string().split(sep_, [&fields](immutable_string &&s) {
        fields.emplace_back(std::move(s));
    });

    // If no names specified, return as list.
    if (names_.empty())
        return data{std::move(fields)};

    // Otherwise, as dictionary.
    if (names_.size() != fields.size())
        throw std::invalid_argument{
            fmt::format("The number of fields must match the number of names ({}), but is {} instead.", names_.size(), fields.size())};

    flat_hash_map<std::string, data> m{};

    for (std::size_t i = 0; i < fields.size(); ++i)
        m.emplace(names_[i], std::move(fields[i]));

    return data{std::move(m)};
}

}  // namespace fairseq2
