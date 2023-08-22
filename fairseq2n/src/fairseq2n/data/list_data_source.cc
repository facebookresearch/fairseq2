// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/list_data_source.h"

#include <cstddef>

namespace fairseq2n::detail {

std::optional<data>
list_data_source::next()
{
    if (iter_ == list_.end())
        return std::nullopt;

    return *iter_++;
}

void
list_data_source::reset()
{
    iter_ = list_.begin();
}

void
list_data_source::record_position(tape &t) const
{
    t.record(iter_ - list_.begin());
}

void
list_data_source::reload_position(tape &t)
{
    iter_ = list_.begin() + t.read<std::ptrdiff_t>();
}

}
