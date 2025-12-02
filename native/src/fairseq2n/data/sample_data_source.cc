// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/sample_data_source.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/core/TransformationHelper.h>

#include "fairseq2n/data/detail/exception.h"
#include "fairseq2n/data/detail/rng.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n::detail {

sample_data_source::sample_data_source(
    std::vector<data_pipeline> &&pipelines,
    std::vector<float32> &&weights,
    std::optional<std::uint64_t> maybe_seed,
    bool allow_repeats)
  : pipelines_(std::move(pipelines)), 
    is_epoch_done_(pipelines_.size()),
    allow_repeats_{allow_repeats}
{
    seed_ = maybe_seed ? *maybe_seed : pseudo_random();

    generator_ = at::make_generator<at::CPUGeneratorImpl>(seed_);

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

    if (!allow_repeats_)
        original_weight_cumsums_ = weight_cumsums_;

    buffer_.reserve(pipelines_.size());

    if (pipelines_.empty())
        finitude_type_ = data_source_finitude_type::finite;
    else {
        auto max_cardinality_pipeline_it =
            std::max_element(pipelines_.begin(), pipelines_.end(),
                             [](const data_pipeline &a, const data_pipeline &b) {
                                 return a.finitude_type() < b.finitude_type();
                             });
        finitude_type_ = max_cardinality_pipeline_it->finitude_type();
    }
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
sample_data_source::reset(bool reset_rng)
{
    buffer_.clear();

    is_epoch_done_.assign(pipelines_.size(), false);

    is_eod_ = false;

    if (reset_rng)
        generator_.set_current_seed(seed_);

    if (!allow_repeats_)
        weight_cumsums_ = original_weight_cumsums_;

    for (data_pipeline &pipeline : pipelines_)
        pipeline.reset(reset_rng);
}

void
sample_data_source::record_position(tape &t, bool strict) const
{
    if (strict) {
        t.record(buffer_);

        t.record(is_epoch_done_);
    }

    t.record(seed_);

    t.record(generator_.get_state());

    t.record(weight_cumsums_);

    for (const data_pipeline &pipeline : pipelines_)
        pipeline.record_position(t, strict);
}

void
sample_data_source::reload_position(tape &t, bool strict)
{
    if (strict) {
        buffer_ = t.read<std::vector<std::optional<data>>>();

        is_epoch_done_ = t.read<std::vector<bool>>();
    } else {
        buffer_.clear();

        is_epoch_done_.assign(pipelines_.size(), false);
    }

    is_eod_ = false;

    seed_ = t.read<std::uint64_t>();

    generator_.set_state(t.read<at::Tensor>());

    weight_cumsums_ = t.read<std::vector<float32>>();

    for (data_pipeline &pipeline : pipelines_)
        pipeline.reload_position(t);
}

data_source_finitude_type
sample_data_source::finitude_type() const noexcept
{
    return finitude_type_;
}

std::size_t
sample_data_source::random_pipeline_index()
{
    auto *gen = generator_.get<at::CPUGeneratorImpl>();

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

    // Binary search finds an index where cumsum[idx] >= sample.
    // When allow_repeats_ is false, exhausted pipelines have cumsum[idx] == cumsum[idx-1].
    // Skip forward to find the next active pipeline that "owns" this probability mass.
    if (!allow_repeats_) { 
        while (lptr < is_epoch_done_.size() && is_epoch_done_[lptr]) lptr++;
    }

    // Should never happen since at least one pipeline is active 
    // (guaranteed by are_all_done() check in next())
    assert(lptr < weight_cumsums_.size());

    return lptr;
}
std::optional<data>
sample_data_source::next_in_pipeline(std::size_t pipeline_idx)
{
    data_pipeline &pipeline = pipelines_[pipeline_idx];

    std::optional<data> maybe_example = pipeline.next();
    if (!maybe_example) {
        is_epoch_done_[pipeline_idx] = true;

        if (allow_repeats_) {
            pipeline.reset();

            // Circle back to the first example.
            maybe_example = pipeline.next();
            if (!maybe_example)
                throw_data_pipeline_error(
                    /*maybe_example=*/std::nullopt, /*recoverable=*/false,
                    "The data pipeline at index {} is empty and cannot be sampled.", pipeline_idx);
        } else
            block(pipeline_idx);
    } else if (pipeline.finitude_type() == data_source_finitude_type::pseudo_infinite)
        is_epoch_done_[pipeline_idx] = true;

    return maybe_example;
}

void
sample_data_source::block(std::size_t idx)
{
    float32 weight = weight_cumsums_[idx];
    if (idx > 0) {
        weight -= weight_cumsums_[idx - 1];
        weight_cumsums_[idx] = weight_cumsums_[idx - 1];
    } else {
        weight_cumsums_[idx] = 0.0F;
    }
    for (std::size_t i = idx + 1; i < weight_cumsums_.size(); ++i) {
        weight_cumsums_[i] -= weight;
    }

    float32 sum = weight_cumsums_.back();

    if (!are_close(sum, 1.0F)) {
        for (float32 &s : weight_cumsums_)
            s /= sum;
    }
}

bool
sample_data_source::are_all_done() noexcept
{
    is_eod_ = std::all_of(is_epoch_done_.begin(), is_epoch_done_.end(), [](bool b) {
        return b;
    });

    return is_eod_;
}

}  // namespace fairseq2n::detail
