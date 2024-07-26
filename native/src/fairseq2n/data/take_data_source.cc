// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/take_data_source.h"

namespace fairseq2n::detail {

std::optional<data>
take_data_source::next()
{
    if (num_examples_read_ == num_examples_)
        return std::nullopt;

    std::optional<data> maybe_example = inner_->next();
    if (maybe_example)
        num_examples_read_++;

    return maybe_example;
}

void
take_data_source::reset(bool reset_rng)
{
    num_examples_read_ = 0;

    inner_->reset(reset_rng);
}

void
take_data_source::record_position(tape &t, bool strict) const
{
    t.record(num_examples_read_);

    inner_->record_position(t, strict);
}

void
take_data_source::reload_position(tape &t, bool strict)
{
    num_examples_read_ = t.read<std::size_t>();

    inner_->reload_position(t, strict);
}

data_source_finitude_type
take_data_source::finitude_type() const noexcept
{
    return data_source_finitude_type::finite;
}

}  // namespace fairseq2n::detail
