// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/text/string_splitter.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "fairseq2n/fmt.h"
#include "fairseq2n/detail/exception.h"

using namespace fairseq2n::detail;

namespace fairseq2n {

string_splitter::string_splitter(
    char separator,
    std::vector<std::string> names,
    std::vector<std::size_t> indices,
    bool exclude)
  : separator_{separator}, names_(std::move(names)), indices_{std::move(indices)}, exclude_{exclude}
{
    if (indices_.empty())
        return;

    if (!names_.empty() && !exclude && names_.size() != indices_.size())
        throw_<std::invalid_argument>(
            "`names` and `indices` must have the same length, but have the lengths {} and {} instead.", names_.size(), indices_.size());

    std::sort(indices_.begin(), indices_.end());
}

data
string_splitter::operator()(data &&d) const
{
    if (!d.is_string())
        throw_<std::invalid_argument>(
            "The input data must be of type `string`, but is of type `{}` instead.", d.type());

    data_list fields{};

    auto idx_pos = indices_.begin();

    std::size_t idx = 0;

    d.as_string().split(separator_, [this, &fields, &idx_pos, &idx](immutable_string &&s) {
        if (idx_pos == indices_.end()) {
            fields.emplace_back(std::move(s));
        } else {
            if (exclude_) {
                if (idx != *idx_pos)
                    fields.emplace_back(std::move(s));
                else
                    ++idx_pos;
            } else {
                if (idx == *idx_pos) {
                    fields.emplace_back(std::move(s));

                    // We got all fields we need, no need to process the rest.
                    if (++idx_pos == indices_.end())
                        return false;
                }
            }
        }

        idx++;

        return true;
    });

    if (idx_pos != indices_.end())
        throw_<std::invalid_argument>(
            "The input string must have at least {} field(s), but has {} instead.", indices_.back(), idx);

    // If no names specified, return as list.
    if (names_.empty())
        return fields;

    // Otherwise, as dictionary.
    if (names_.size() != fields.size())
        throw_<std::invalid_argument>(
            "The number of fields must match the number of names ({}), but is {} instead.", names_.size(), fields.size());

    data_dict dict{};

    for (std::size_t i = 0; i < fields.size(); ++i)
        dict.emplace(names_[i], std::move(fields[i]));

    return dict;
}

}  // namespace fairseq2n
