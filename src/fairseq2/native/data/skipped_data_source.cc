// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/skipped_data_source.h"

namespace fairseq2::detail {

std::optional<data>
skipped_data_source::next()
{
    if (!skipped_) {
        for (std::size_t i = 0; i < num_examples_; i++)
            if (!inner_->next())
                break;

        skipped_ = true;
    }

    return inner_->next();
}

void
skipped_data_source::reset()
{
    skipped_ = false;

    inner_->reset();
}

void
skipped_data_source::record_position(tape &t) const
{
    t.record(skipped_);

    inner_->record_position(t);
}

void
skipped_data_source::reload_position(tape &t)
{
    skipped_ = t.read<bool>();

    inner_->reload_position(t);
}

}  // namespace fairseq2::detail
