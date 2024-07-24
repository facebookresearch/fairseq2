// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/skip_data_source.h"

namespace fairseq2n::detail {

std::optional<data>
skip_data_source::next()
{
    if (num_examples_ == 0)
        return inner_->next();

    if (!skip_) {
        for (std::size_t i = 0; i < num_examples_; i++)
            if (!inner_->next())
                break;

        skip_ = true;
    }

    return inner_->next();
}

void
skip_data_source::reset(bool reset_rng)
{
    skip_ = false;

    inner_->reset(reset_rng);
}

void
skip_data_source::record_position(tape &t, bool strict) const
{
    t.record(skip_);

    inner_->record_position(t, strict);
}

void
skip_data_source::reload_position(tape &t, bool strict)
{
    skip_ = t.read<bool>();

    inner_->reload_position(t, strict);
}

data_source_finitude_type
skip_data_source::finitude_type() const noexcept
{
    return inner_->finitude_type();
}

}  // namespace fairseq2n::detail
