// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/composite_data_source.h"

namespace fairseq2n::detail {

composite_data_source::composite_data_source(
    std::vector<data_pipeline> &&pipelines,
    index_generator_fn &&index_gen_fn,
    bool stop_at_shortest)
  : pipelines_(std::move(pipelines)),
    next_index_gen_{std::move(index_gen_fn)},
    stop_at_shortest_{stop_at_shortest}
{
    if (!stop_at_shortest) {
        is_epoch_done_ = std::vector<bool>(pipelines_.size(), false);
        buffer_ = std::vector<std::optional<data>>(pipelines_.size(), std::nullopt);
    }
}

std::optional<data>
composite_data_source::next()
{
    if (stop_at_shortest_) // with this flag on, the operator is a simple iterator
        return pipelines_[next_index_gen_()].next();

    // One or more data pipelines might be empty, so we have to keep looping
    std::optional<data> output{};
    while (!output && !eod()) {
        auto pipeline_idx = next_index_gen_();
        auto &maybe_example = buffer_[pipeline_idx];

        if (!maybe_example) // init buffer at first call
            maybe_example = next_in_pipeline(pipeline_idx);

        output = std::exchange(maybe_example, next_in_pipeline(pipeline_idx));
    }

    return output;
}

void
composite_data_source::reset()
{
    for (data_pipeline &pipeline : pipelines_)
        pipeline.reset();

    if (!stop_at_shortest_) {
        buffer_.assign(pipelines_.size(), std::nullopt);
        is_epoch_done_.assign(pipelines_.size(), false);
        is_eod_ = false;
    }
}

void
composite_data_source::record_position(tape &t) const
{
    for (const data_pipeline &pipeline : pipelines_)
        pipeline.record_position(t);

    if (!stop_at_shortest_) {
        t.record(buffer_);
        t.record(is_epoch_done_);
    }
}

void
composite_data_source::reload_position(tape &t)
{
    for (data_pipeline &pipeline : pipelines_)
        pipeline.reload_position(t);

    if (!stop_at_shortest_) {
        buffer_ = t.read<std::vector<std::optional<data>>>();
        is_epoch_done_ = t.read<std::vector<bool>>();
        is_eod_ = false;
    }
}

std::optional<data>
composite_data_source::next_in_pipeline(std::size_t pipeline_idx)
{
    data_pipeline &pipeline = pipelines_[pipeline_idx];

    std::optional<data> maybe_example = pipeline.next();
    if (!maybe_example) {
        is_epoch_done_[pipeline_idx] = true;

        pipeline.reset();

        // Circle back to the first example.
        maybe_example = pipeline.next();
    }

    return maybe_example;
}

bool
composite_data_source::eod()
{
    is_eod_ = is_eod_ || std::all_of(
        is_epoch_done_.begin(), is_epoch_done_.end(), [](bool b)
        {
            return b;
        });

    return is_eod_;
}

}  // namespace fairseq2n::detail
