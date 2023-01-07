// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/sharded_data_source.h"

namespace fairseq2::detail {

std::optional<data>
sharded_data_source::next()
{
    std::size_t skip_f = shard_idx_;
    std::size_t skip_l = num_shards_ - shard_idx_ - 1;

    if (!skip_inner(skip_f))
        return {};

    std::optional<data> d = inner_->next();
    if (!d)
        return {};

    if (!skip_inner(skip_l))
        return {};

    return d;
}

std::size_t
sharded_data_source::skip(std::size_t num_examples)
{
    return inner_->skip(num_examples * num_shards_) / num_shards_;
}

void
sharded_data_source::reset()
{
    inner_->reset();
}

void
sharded_data_source::record_position(tape &t) const
{
    inner_->record_position(t);
}

void
sharded_data_source::reload_position(tape &t)
{
    inner_->reload_position(t);
}

bool
sharded_data_source::skip_inner(std::size_t num_examples)
{
    return inner_->skip(num_examples) == num_examples;
}

}  // namespace fairseq2::detail
