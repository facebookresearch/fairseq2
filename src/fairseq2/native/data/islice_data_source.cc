// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "fairseq2/native/data/islice_data_source.h"


namespace fairseq2::detail {

std::optional<data>
islice_data_source::next()
{
    if (next_index_ == 0 && start_ != 0)
        next_index_ += inner_->skip(start_);
    if ((next_index_ - start_) % step_ == 1)
        next_index_ += inner_->skip(step_ - 1);

    // return only if haven't reached stop
    if (!stop_.has_value() || next_index_ < stop_.value()) {
        next_index_++;
        return inner_->next();
    }

    return {};
}

std::size_t
islice_data_source::skip(std::size_t num_examples)
{
    std::size_t skipped = 0;
    while (skipped < num_examples && next().has_value())
        skipped++;

    return skipped;
}

void
islice_data_source::reset()
{
    next_index_ = 0;

    inner_->reset();
}

void
islice_data_source::record_position(tape &t) const
{
    t.record(next_index_);

    inner_->record_position(t);
}

void
islice_data_source::reload_position(tape &t)
{
    next_index_ = t.read<std::size_t>();

    inner_->reload_position(t);
}

} // namespace fairseq2::detail
