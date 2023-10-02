// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/sample_data_source.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/Functions.h>

#include <stdexcept>

#include "fairseq2n/utils/tensor.h"

namespace fairseq2n::detail {

sample_data_source::sample_data_source(std::vector<data_pipeline> &&pipelines, std::vector<float32> &&weights, bool stop_at_shortest)
    : pipelines_(std::move(pipelines))
{
    weights_ = make_tensor_from_vector(weights, { static_cast<std::int64_t>(pipelines_.size()) });
    generator_ = at::globalContext().defaultGenerator(c10::DeviceType::CPU);
}

std::optional<data>
sample_data_source::next()
{
    if (eod_)
        return std::nullopt;

    std::optional<data> output = pipelines_[next_index()].next();
    if (!output)
        eod_ = true;

    return output;
}

void
sample_data_source::reset()
{
    eod_ = false;
    for (data_pipeline &p : pipelines_)
        p.reset();
}

void
sample_data_source::record_position(tape &t) const
{
    t.record(eod_);
    for (const data_pipeline &p : pipelines_)
        p.record_position(t);
}

void
sample_data_source::reload_position(tape &t)
{
    eod_ = t.read<bool>();
    for (data_pipeline &p : pipelines_)
        p.reload_position(t);
}

std::size_t
sample_data_source::next_index()
{
    auto result = at::multinomial(weights_, 1, false, generator_)
        .item<std::int64_t>();

    return static_cast<std::size_t>(result);
}

}  // namespace fairseq2::detail
