// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/list_data_source.h"

#include <algorithm>
#include <cstddef>

#include "fairseq2/native/utils/cast.h"

namespace fairseq2::detail {

std::optional<data>
list_data_source::next()
{
    if (iter_ == list_.end())
        return {};

    return *iter_++;
}

std::size_t
list_data_source::skip(std::size_t num_examples)
{
    auto num_remaining = static_cast<std::size_t>(list_.end() - iter_);

    std::size_t n = std::min(num_examples, num_remaining);

    iter_ += static_cast<std::vector<data>::difference_type>(n);

    return n;
}

void
list_data_source::reset()
{
    iter_ = list_.begin();
}

void
list_data_source::record_position(tape &t) const
{
    t.record(conditional_cast<std::int64_t>(iter_ - list_.begin()));
}

void
list_data_source::reload_position(tape &t)
{
    auto offset = t.read<std::vector<data>::difference_type>();

    tape::check(offset >= 0);
    tape::check(offset <= ssize(list_));

    iter_ = list_.begin() + offset;
}

}
