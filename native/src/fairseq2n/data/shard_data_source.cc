// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/shard_data_source.h"

namespace fairseq2n::detail {

std::optional<data>
shard_data_source::next()
{
    if (num_shards_ == 1)
        return inner_->next();

    for (std::size_t i = 0; i < shard_idx_; i++)
        if (!inner_->next())
            return std::nullopt;

    std::optional<data> maybe_example = inner_->next();
    if (!maybe_example)
        return std::nullopt;

    for (std::size_t i = 0; i < num_shards_ - shard_idx_ - 1; i++)
        if (!inner_->next())
            return std::nullopt;

    return maybe_example;
}

void
shard_data_source::reset(bool reset_rng)
{
    inner_->reset(reset_rng);
}

void
shard_data_source::record_position(tape &t, bool strict) const
{
    inner_->record_position(t, strict);
}

void
shard_data_source::reload_position(tape &t, bool strict)
{
    inner_->reload_position(t, strict);
}

bool
shard_data_source::is_infinite() const noexcept
{
    return inner_->is_infinite();
}

}  // namespace fairseq2n::detail
