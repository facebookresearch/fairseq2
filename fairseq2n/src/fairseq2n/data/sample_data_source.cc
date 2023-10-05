// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/sample_data_source.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/Functions.h>

#include "fairseq2n/utils/tensor.h"

namespace fairseq2n::detail {

sample_data_source::sample_data_source(std::vector<data_pipeline> &&pipelines, std::vector<float32> &&weights, bool stop_at_shortest)
{
    weights_ = make_tensor_from_vector(weights, { static_cast<std::int64_t>(pipelines.size()) });
    generator_ = at::globalContext().defaultGenerator(c10::DeviceType::CPU);

    auto gen = [this]()
    {
        auto result = at::multinomial(weights_, 1, false, generator_)
            .item<std::int64_t>();

        return static_cast<std::size_t>(result);
    };

    inner_ = std::make_unique<composite_data_source>(std::move(pipelines), std::move(gen), stop_at_shortest);
}

std::optional<data>
sample_data_source::next()
{
    auto output = inner_->next();
    if (!output)
        return std::nullopt;

    return output;
}

void
sample_data_source::reset()
{
    inner_->reset();
}

void
sample_data_source::record_position(tape &t) const
{
    inner_->record_position(t);
}

void
sample_data_source::reload_position(tape &t)
{
    inner_->reload_position(t);
}

}  // namespace fairseq2::detail
