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
    
    
}

void cat_data_source::reset()
{
    pipeline1_ = {};
    pipeline2_ = {};
}  

void cat_data_source::record_position(tape &t) const
{
    t.record(pipeline1_);
    t.record(pipeline2_);
}

void cat_data_source::reload_position(tape &t)
{

    
}

} // namespace fairseq2n::detail