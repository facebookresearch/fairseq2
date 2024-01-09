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
    if (!skip_) {
        for (std::size_t i = 0; i < num_examples_; i++)
            if (!inner_->next())
                break;

        skip_ = true;
    }

    return inner_->next();
}

void
skip_data_source::reset()
{
    skip_ = false;

    inner_->reset();
}

void
skip_data_source::record_position(tape &t) const
{
    t.record(skip_);

    inner_->record_position(t);
}

void
skip_data_source::reload_position(tape &t)
{
    skip_ = t.read<bool>();

    inner_->reload_position(t);
}

bool
skip_data_source::is_infinite() const noexcept
{
    return inner_->is_infinite();
}

}  // namespace fairseq2n::detail
