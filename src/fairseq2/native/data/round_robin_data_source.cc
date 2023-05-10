// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/round_robin_data_source.h"
#include <vector>

namespace fairseq2::detail {


std::optional<data>
round_robin_data_source::next()
{
    if (all_datasources_done())
        return {};

    std::optional<data> result = data_pipelines_[index_].next();
    if (!result) {
        reset_pipeline(index_);
        result = data_pipelines_[index_].next();
    }
    index_ = (index_ + 1) % pipelines_count_;

    return result;
}

std::size_t
round_robin_data_source::skip(std::size_t num_examples)
{
    if (all_datasources_done())
        return 0;

    auto num_examples_per_pipeline = get_split_by_pipeline(num_examples);
    for (std::size_t i = 0; i < pipelines_count_; ++i)
    {
        skip_elements_for_pipeline(i, num_examples_per_pipeline[i]);
    }
    index_ = (index_ + num_examples) % pipelines_count_;

    return num_examples;
}

void
round_robin_data_source::reset()
{
    index_ = 0;
    epoch_done_ = std::vector<bool>(pipelines_count_, false);

    for (auto &dp : data_pipelines_)
        dp.reset();
}

void
round_robin_data_source::record_position(tape &t) const
{
    t.record(index_);
    t.record(epoch_done_);
    for (auto &dp : data_pipelines_)
        dp.record_position(t);
}

void
round_robin_data_source::reload_position(tape &t)
{
    index_ = t.read<std::size_t>();
    epoch_done_ = t.read<std::vector<bool>>();
    for (auto &dp : data_pipelines_)
        dp.reload_position(t);
}

// Helper methods

/// @brief splits input value on different datasources equally
/// @param num_examples value to split equally on different datasources
/// @return a vector containing the num_examples for each datasource
/// @example to split 23 on 4 datasources would result in 6, 6, 6, 5
std::vector<std::size_t>
round_robin_data_source::get_split_by_pipeline(std::size_t num_examples) const
{
    std::vector<std::size_t> num_examples_per_pipeline(pipelines_count_, num_examples / pipelines_count_);
    const auto remaining = num_examples % pipelines_count_;
    for (std::size_t i = 0; i < remaining; ++i)
    {
        num_examples_per_pipeline[(index_ + i) % pipelines_count_] += 1;
    }

    // TODO We can avoid copy here but I was having an exception when using std::move -- Still not familiar with the syntax
    return num_examples_per_pipeline;
}

void
round_robin_data_source::skip_elements_for_pipeline(std::size_t pipeline_index, std::size_t num_examples)
{
    std::size_t remaining = num_examples;
    while (remaining != 0)
    {
        auto skipped = data_pipelines_[pipeline_index].skip(remaining);
        if (skipped < remaining)
            reset_pipeline(pipeline_index);
        remaining -= skipped;
    }
}

void
round_robin_data_source::reset_pipeline(std::size_t pipeline_index)
{
    epoch_done_[pipeline_index] = true;
    data_pipelines_[pipeline_index].reset();
}

bool
round_robin_data_source::all_datasources_done()
{
    return std::all_of(epoch_done_.begin(), epoch_done_.end(), [](bool v) { return v; });
}

}  // namespace fairseq2::detail
