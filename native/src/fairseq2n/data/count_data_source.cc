// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/count_data_source.h"

#include "fairseq2n/data/data.h"

namespace fairseq2n::detail {

std::optional<data>
count_data_source::next()
{
    std::int64_t output = counter_;

    counter_ += step_;

    if (key_)
        return data_dict{{*key_, output}};

    return output;
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

bool
count_data_source::is_infinite() const noexcept
{
    return true;
}

}
