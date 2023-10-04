// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/cat_data_source.h"
#include <vector>

namespace fairseq2n::detail {

cat_data_source::cat_data_source(
    std::vector<std::reference_wrapper<data_pipeline>> &&pipeline1,
    std::vector<std::reference_wrapper<data_pipeline>> &&pipeline2)
    : pipeline1_{std::move(pipeline1)}
    , pipeline2_{std::move(pipeline2)}
{
}



std::optional<data>
cat_data_source::next()
{
    if (pipeline1_idx_ < pipeline1_.size()) {
        auto &pipeline = pipeline1_[pipeline1_idx_];
        auto next = pipeline.get().next();
        if (next) return next;
        pipeline1_idx_ += 1;
    }
    if (pipeline2_idx_ < pipeline2_.size()) {
        auto &pipeline = pipeline2_[pipeline2_idx_];
        auto next = pipeline.get().next();
        if (next) return next;
        pipeline2_idx_ += 1;
    }
    return std::nullopt;
}

void cat_data_source::reset()
{
    pipeline1_ = 0;
    pipeline2_ = 0;
}  

void cat_data_source::record_position(tape &t) const
{
    t.record(pipeline1_);
    t.record(pipeline2_);
}

void cat_data_source::reload_position(tape &t)
{
    t.reload(pipeline1_);
    t.reload(pipeline2_);
}

} // namespace fairseq2n::detail