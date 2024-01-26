// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/sample_data_source.h"

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/core/TransformationHelper.h>

#include "fairseq2n/data/detail/exception.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n::detail {

sample_data_source::sample_data_source(
    std::vector<data_pipeline> &&pipelines, std::vector<float32> &&weights)
  : pipelines_(std::move(pipelines)), is_epoch_done_(pipelines_.size())
{
    generator_ = at::globalContext().defaultGenerator(at::kCPU);

    weight_cumsums_.reserve(weights.size());

    float32 sum = 0.0F;

    for (float32 weight : weights) {
        sum += weight;

        weight_cumsums_.push_back(sum);
    }

    if (!are_close(sum, 1.0F)) {
        // Normalize the cumulative probability distribution.
        for (float32 &s : weight_cumsums_)
            s /= sum;
    }

    buffer_.reserve(pipelines_.size());

    is_infinite_ = std::all_of(
        pipelines_.begin(), pipelines_.end(), [](const data_pipeline &p)
        {
            return p.is_infinite();
        });
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

    return std::exchange(buffer_[pipeline_idx], next_in_pipeline(pipeline_idx));
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

bool
sample_data_source::is_infinite() const noexcept
{
    return is_infinite_;
}

std::size_t
sample_data_source::random_pipeline_index()
{
    std::lock_guard<std::mutex> guard{generator_.mutex()};

    auto *gen = at::check_generator<at::CPUGeneratorImpl>(generator_);

    float32 sample = at::transformation::uniform_real(gen->random(), 0.0F, 1.0F);

    std::size_t lptr = 0;
    std::size_t rptr = weight_cumsums_.size() - 1;

    while (rptr - lptr > 0) {
        std::size_t mptr = lptr + (rptr - lptr) / 2;

        if (float32 sum = weight_cumsums_[mptr]; sum < sample)
            lptr = mptr + 1;
        else
            rptr = mptr;
    }

    return lptr;
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
    } else if (pipeline.is_infinite())
        is_epoch_done_[pipeline_idx] = true;

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

}  // namespace fairseq2n::detail
