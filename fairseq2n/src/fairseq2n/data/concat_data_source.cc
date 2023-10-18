// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/concat_data_source.h"
#include <vector>

namespace fairseq2n::detail {

concat_data_source::concat_data_source(std::vector<data_pipeline> &&pipelines)
    : pipelines_(std::move(pipelines))
{}

std::optional<data>
concat_data_source::next()
{
    std::optional<data> d;
    for (auto &p : pipelines_) {
        d = p.next();
        if (d)
          return d;
    }
    return {};
}

void concat_data_source::reset()
{
    for (auto &pipeline : pipelines_)
        pipeline.reset();
}  

void concat_data_source::record_position(tape &t) const
{
    for (auto &pipeline : pipelines_)
        pipeline.record_position(t);
}

void concat_data_source::reload_position(tape &t)
{
    for (auto &pipeline : pipelines_)
        pipeline.reload_position(t);
}

} // namespace fairseq2n::detail

