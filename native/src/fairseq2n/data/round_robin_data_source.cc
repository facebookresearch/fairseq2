// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/round_robin_data_source.h"

#include <algorithm>

namespace fairseq2n::detail {

round_robin_data_source::round_robin_data_source(
    std::vector<data_pipeline> &&pipelines, bool stop_at_shortest)
  : pipelines_(std::move(pipelines)),
    is_epoch_done_(pipelines_.size()),
    stop_at_shortest_{stop_at_shortest}
{
    buffer_.reserve(pipelines_.size());

    is_infinite_ = std::all_of(
        pipelines_.begin(), pipelines_.end(), [](const data_pipeline &p)
        {
            return p.is_infinite();
        });
}

std::optional<data>
round_robin_data_source::next()
{
    if (pipelines_.empty() || is_eod_)
        return std::nullopt;

    // At the beginning of the next round, check if all data pipelines had at
    // least one epoch. If that is the case, we can signal EOD.
    if (buffer_idx_ == 0) {
        if (are_all_done())
            return std::nullopt;
    }

    // If this is the first call, gather the first round of examples.
    if (buffer_.empty()) {
        for (std::size_t i = 0; i < pipelines_.size(); i++)
            buffer_.push_back(next_in_pipeline(i));
    }

    std::optional<data> output{};

    // One or more data pipelines might be empty, so we have to check if a
    // buffered example has a value before returning it.
    for (; !output && buffer_idx_ < buffer_.size(); buffer_idx_++) {
        std::optional<data> &maybe_example = buffer_[buffer_idx_];
        if (maybe_example)
            // Fill the position with the next round's example.
            output = std::exchange(maybe_example, next_in_pipeline(buffer_idx_));
    }

    if (buffer_idx_ == buffer_.size())
        buffer_idx_ = 0;

    // Might not have a value if all data pipelines were empty.
    return output;
}

void
round_robin_data_source::reset()
{
    buffer_.clear();

    buffer_idx_ = 0;

    is_epoch_done_.assign(pipelines_.size(), false);

    is_eod_ = false;

    for (data_pipeline &pipeline : pipelines_)
        pipeline.reset();
}

void
round_robin_data_source::record_position(tape &t) const
{
    t.record(buffer_);

    t.record(buffer_idx_);

    t.record(is_epoch_done_);

    for (const data_pipeline &pipeline : pipelines_)
        pipeline.record_position(t);
}

void
round_robin_data_source::reload_position(tape &t)
{
    buffer_ = t.read<std::vector<std::optional<data>>>();

    buffer_idx_ = t.read<std::size_t>();

    is_epoch_done_ = t.read<std::vector<bool>>();

    is_eod_ = false;

    for (data_pipeline &pipeline : pipelines_)
        pipeline.reload_position(t);
}

bool
round_robin_data_source::is_infinite() const noexcept
{
    return is_infinite_;
}

std::optional<data>
round_robin_data_source::next_in_pipeline(std::size_t pipeline_idx)
{
    data_pipeline &pipeline = pipelines_[pipeline_idx];

    std::optional<data> maybe_example = pipeline.next();
    if (!maybe_example) {
        is_epoch_done_[pipeline_idx] = true;

        pipeline.reset();

        // Circle back to the first example.
        maybe_example = pipeline.next();
    } else if (pipeline.is_infinite())
        is_epoch_done_[pipeline_idx] = true;

    return maybe_example;
}

bool
round_robin_data_source::are_all_done() noexcept
{
    if (stop_at_shortest_) {
        is_eod_ = std::any_of(
            is_epoch_done_.begin(), is_epoch_done_.end(), [](bool b)
            {
                return b;
            });
    } else {
        is_eod_ = std::all_of(
            is_epoch_done_.begin(), is_epoch_done_.end(), [](bool b)
            {
                return b;
            });
    }

    return is_eod_;
}

}  // namespace fairseq2n::detail
