// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/repeat_data_source.h"

#include "fairseq2n/data/data.h"

namespace fairseq2n::detail {

std::optional<data>
repeat_data_source::next()
{
    if (num_repeats_ && *num_repeats_ == 0)
        return std::nullopt;

    while (true) {
        std::optional<data> maybe_example = inner_->next();
        if (maybe_example) {
            has_data_ = true;

            return maybe_example;
        }

        if (!has_data_)
            return std::nullopt;

        if (num_repeats_ && repeat_nr_ == *num_repeats_ - 1)
            return std::nullopt;

        repeat_nr_++;

        inner_->reset(reset_rng_);
    }
}

void
repeat_data_source::reset(bool reset_rng)
{
    has_data_ = false;

    repeat_nr_ = 0;

    inner_->reset(reset_rng);
}

void
repeat_data_source::record_position(tape &t, bool strict) const
{
    t.record(repeat_nr_);

    inner_->record_position(t, strict);
}

void
repeat_data_source::reload_position(tape &t, bool strict)
{
    repeat_nr_ = t.read<std::size_t>();

    inner_->reload_position(t, strict);
}

data_source_finitude_type
repeat_data_source::finitude_type() const noexcept
{
    return (num_repeats_ ? inner_->finitude_type() : data_source_finitude_type::infinite);
}

}
