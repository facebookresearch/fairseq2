// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/round_robin_data_source.h"

namespace fairseq2n::detail {

round_robin_data_source::round_robin_data_source(std::vector<data_pipeline> &&pipelines, bool stop_at_shortest)
{
    pipelines_count_ = pipelines.size();
    pipeline_idx_ = 0;

    auto gen = [this]()
    {
        this->pipeline_idx_ %= pipelines_count_;

        return this->pipeline_idx_++;
    };

    circular_ = std::make_unique<circular_data_source>(std::move(pipelines), std::move(gen), stop_at_shortest);
}


std::optional<data>
round_robin_data_source::next()
{
    auto output = circular_->next();
    if (!output)
        return std::nullopt;

    return output;
}

void
round_robin_data_source::reset()
{
    circular_->reset();

    pipeline_idx_ = 0;
}

void
round_robin_data_source::record_position(tape &t) const
{
    circular_->record_position(t);

    t.record(pipeline_idx_);
}

void
round_robin_data_source::reload_position(tape &t)
{
    circular_->reload_position(t);

    pipeline_idx_ = t.read<std::size_t>();
}

}  // namespace fairseq2n::detail
