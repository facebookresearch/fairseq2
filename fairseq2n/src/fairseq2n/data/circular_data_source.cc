// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/circular_data_source.h"

namespace fairseq2n::detail {

circular_data_source::circular_data_source(std::vector<data_pipeline> &&pipelines, index_generator_fn &&index_gen_fn)
    : pipelines_(std::move(pipelines)), next_index_gen_(std::move(index_gen_fn))
{
    is_epoch_done_ = std::vector<bool>(pipelines_.size(), false);
    buffer_ = std::vector<std::optional<data>>(pipelines_.size(), std::nullopt);
}

std::optional<data>
circular_data_source::next()
{
    if (eod())
        return {};

    auto pipeline_idx = next_index_gen_();

    if (!buffer_[pipeline_idx]) // init buffer at index
        buffer_[pipeline_idx] = next_in_pipeline(pipeline_idx);

    auto output = buffer_[pipeline_idx];
    buffer_[pipeline_idx] = next_in_pipeline(pipeline_idx);

    return output;
}

void
circular_data_source::reset()
{
    buffer_.clear();

    is_epoch_done_.assign(pipelines_.size(), false);

    is_eod_ = false;

    for (data_pipeline &pipeline : pipelines_)
        pipeline.reset();
}

void
circular_data_source::record_position(tape &t) const
{
    t.record(buffer_);

    t.record(is_epoch_done_);

    for (const data_pipeline &pipeline : pipelines_)
        pipeline.record_position(t);
}

void
circular_data_source::reload_position(tape &t)
{
    buffer_ = t.read<std::vector<std::optional<data>>>();

    is_epoch_done_ = t.read<std::vector<bool>>();

    is_eod_ = false;

    for (data_pipeline &pipeline : pipelines_)
        pipeline.reload_position(t);
}

std::optional<data>
circular_data_source::next_in_pipeline(std::size_t pipeline_idx)
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
circular_data_source::eod()
{
    is_eod_ = is_eod_ || std::all_of(
        is_epoch_done_.begin(), is_epoch_done_.end(), [](bool b)
        {
            return b;
        });

    return is_eod_;
}

}  // namespace fairseq2n::detail