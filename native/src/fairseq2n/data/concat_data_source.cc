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
    is_infinite_ = std::any_of(
        pipelines_.begin(), pipelines_.end(), [](const data_pipeline &p)
        {
            return p.is_infinite();
        });
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

void concat_data_source::reset()
{
    for (data_pipeline &pipeline : pipelines_)
        pipeline.reset();
}

void concat_data_source::record_position(tape &t) const
{
    for (const data_pipeline &pipeline : pipelines_)
        pipeline.record_position(t);
}

void concat_data_source::reload_position(tape &t)
{
    for (data_pipeline &pipeline : pipelines_)
        pipeline.reload_position(t);
}

bool
concat_data_source::is_infinite() const noexcept
{
    return is_infinite_;
}

} // namespace fairseq2n::detail
