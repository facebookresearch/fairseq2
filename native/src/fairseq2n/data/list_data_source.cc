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
    if (pos_ == list_.end())
        return std::nullopt;

    return *pos_++;
}

void
list_data_source::reset()
{
    pos_ = list_.begin();
}

void
list_data_source::record_position(tape &t) const
{
    t.record(pos_ - list_.begin());
}

void
list_data_source::reload_position(tape &t)
{
    pos_ = list_.begin() + t.read<std::ptrdiff_t>();
}

bool
list_data_source::is_infinite() const noexcept
{
    return false;
}

}
