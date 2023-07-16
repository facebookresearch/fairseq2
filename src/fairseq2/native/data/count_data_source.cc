// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/count_data_source.h"

namespace fairseq2::detail {

std::optional<data>
count_data_source::next()
{
    return counter_++;
}

void
count_data_source::reset()
{
    counter_ = start_;
}

void
count_data_source::record_position(tape &t) const
{
    t.record(counter_);
}

void
count_data_source::reload_position(tape &t)
{
    counter_ = t.read<std::int64_t>();
}

}
