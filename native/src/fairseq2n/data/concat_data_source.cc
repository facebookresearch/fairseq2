// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/concat_data_source.h"

#include <algorithm>

namespace fairseq2n::detail {

concat_data_source::concat_data_source(std::vector<data_pipeline> &&pipelines) noexcept
  : pipelines_(std::move(pipelines))
{
    finitude_type_ = std::max_element(
        pipelines_.begin(), pipelines_.end(), [](const data_pipeline &a, const data_pipeline &b)
        {
            return a.is_infinite() < b.is_infinite();
        })->is_infinite();
}

std::optional<data>
concat_data_source::next()
{
    for (data_pipeline &pipeline : pipelines_) {
        if (std::optional<data> maybe_example = pipeline.next())
            return maybe_example;
    }

    return std::nullopt;
}

void concat_data_source::reset(bool reset_rng)
{
    for (data_pipeline &pipeline : pipelines_)
        pipeline.reset(reset_rng);
}

void concat_data_source::record_position(tape &t, bool strict) const
{
    for (const data_pipeline &pipeline : pipelines_)
        pipeline.record_position(t, strict);
}

void concat_data_source::reload_position(tape &t, bool)
{
    for (data_pipeline &pipeline : pipelines_)
        pipeline.reload_position(t);
}

data_source_finitude_type
concat_data_source::is_infinite() const noexcept
{
    return finitude_type_;
}

} // namespace fairseq2n::detail
