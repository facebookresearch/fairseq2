// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/cat_data_source.h"
#include <vector>

namespace fairseq2n::detail {

cat_data_source::cat_data_source(
    std::vector<data_pipeline> &&pipeline1,
    std::vector<data_pipeline> &&pipeline2)
    : pipeline1_{std::move(pipeline1)}
    , pipeline2_{std::move(pipeline2)}
{}

std::optional<data>
cat_data_source::next()
{
    std::optional<data> d;
    for (auto &p : pipeline1_) {
        d = p.next();
        if (d) {
            return d;
        }
    }
    for (auto &p : pipeline2_) {
        d = p.next();
        if (d) {
            return d;
        }
    }
    return {};
}

void cat_data_source::reset()
{
    for (auto &pipeline : pipeline1_)
        pipeline.reset();
    for (auto &pipeline : pipeline2_)
        pipeline.reset();
}  

void cat_data_source::record_position(tape &t) const
{
    for (auto &pipeline : pipeline1_)
        pipeline.record_position(t);
    for (auto &pipeline : pipeline2_)
        pipeline.record_position(t);
}

void cat_data_source::reload_position(tape &t)
{
    for (auto &pipeline : pipeline1_)
        pipeline.reload_position(t);
    for (auto &pipeline : pipeline2_)
        pipeline.reload_position(t);
}

std::vector<data_pipeline> cat_data_source::concatenate(
    std::vector<data_pipeline> &&pipeline1,
    std::vector<data_pipeline> &&pipeline2)
{
    std::vector<data_pipeline> result;
    result.reserve(pipeline1.size() + pipeline2.size());
    for (auto &&pipeline : pipeline1) {
        result.push_back(std::move(pipeline));
    }
    for (auto &&pipeline : pipeline2) {
        result.push_back(std::move(pipeline));
    }
    return result;
}
} // namespace fairseq2n::detail
