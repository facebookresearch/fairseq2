// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/ranged_data_source.h"

namespace fairseq2::detail {

std::optional<data>
ranged_data_source::next()
{
    if (num_examples_read_ == num_examples_)
        return std::nullopt;

    std::optional<data> d = inner_->next();
    if (d)
        num_examples_read_++;

    return d;
}

void
ranged_data_source::reset()
{
    num_examples_read_ = 0;

    inner_->reset();
}

void
ranged_data_source::record_position(tape &t) const
{
    t.record(num_examples_read_);

    inner_->record_position(t);
}

void
ranged_data_source::reload_position(tape &t)
{
    num_examples_read_ = t.read<std::size_t>();

    inner_->reload_position(t);
}

}  // namespace fairseq2::detail
