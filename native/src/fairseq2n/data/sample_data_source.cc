// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/sample_data_source.h"

#include <stdexcept>

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/Functions.h>

#include "fairseq2n/data/detail/exception.h"
#include "fairseq2n/detail/exception.h"
#include "fairseq2n/utils/tensor.h"

namespace fairseq2n::detail {

sample_data_source::sample_data_source(
    std::vector<data_pipeline> &&pipelines, std::vector<float32> &&weights)
  : pipelines_(std::move(pipelines)), is_epoch_done_(pipelines_.size())
{
    weights_ = make_tensor_from_vector(weights, { static_cast<std::int64_t>(pipelines_.size()) });

    generator_ = at::globalContext().defaultGenerator(at::kCPU);

    buffer_.reserve(pipelines_.size());
}

std::optional<data>
sample_data_source::next()
{
    if (pipelines_.empty() || are_all_done())
        return std::nullopt;

    // If this is the first call, gather the first round of examples.
    if (buffer_.empty()) {
        for (std::size_t i = 0; i < pipelines_.size(); i++)
            buffer_.push_back(next_in_pipeline(i));
    }

    std::size_t pipeline_idx = random_pipeline_index();

    data example = std::exchange(buffer_[pipeline_idx], next_in_pipeline(pipeline_idx));

    return std::move(example);
}

void
sample_data_source::reset()
{
    buffer_.clear();

    is_epoch_done_.assign(pipelines_.size(), false);

    is_eod_ = false;

    for (data_pipeline &pipeline : pipelines_)
        pipeline.reset();
}

void
sample_data_source::record_position(tape &t) const
{
    t.record(buffer_);

    t.record(is_epoch_done_);

    for (const data_pipeline &pipeline : pipelines_)
        pipeline.record_position(t);
}

void
sample_data_source::reload_position(tape &t)
{
    buffer_ = t.read<std::vector<data>>();

    is_epoch_done_ = t.read<std::vector<bool>>();

    is_eod_ = false;

    for (data_pipeline &pipeline : pipelines_)
        pipeline.reload_position(t);
}

std::size_t
sample_data_source::random_pipeline_index()
{
    auto result = at::multinomial(weights_, 1, false, generator_).item<std::int64_t>();

    return static_cast<std::size_t>(result);
}

data
sample_data_source::next_in_pipeline(std::size_t pipeline_idx)
{
    data_pipeline &pipeline = pipelines_[pipeline_idx];

    std::optional<data> maybe_example = pipeline.next();
    if (!maybe_example) {
        is_epoch_done_[pipeline_idx] = true;

        pipeline.reset();

        // Circle back to the first example.
        maybe_example = pipeline.next();
        if (!maybe_example)
            throw_data_pipeline_error(/*maybe_example=*/std::nullopt, /*recoverable=*/false,
                "The data pipeline at index {} is empty and cannot be sampled.", pipeline_idx);
    }

    return std::move(*maybe_example);
}

bool
sample_data_source::are_all_done() noexcept
{
    is_eod_ = std::all_of(
        is_epoch_done_.begin(), is_epoch_done_.end(), [](bool b)
        {
            return b;
        });

    return is_eod_;
}

}  // namespace fairseq2::detail
