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

    if (maybe_key_)
        return data_dict{{*maybe_key_, output}};

    return output;
}

void
count_data_source::reset(bool)
{
    counter_ = start_;
}

void
count_data_source::record_position(tape &t, bool) const
{
    t.record(counter_);
}

void
count_data_source::reload_position(tape &t, bool)
{
    counter_ = t.read<std::int64_t>();
}

data_source_finitude_type
count_data_source::finitude_type() const noexcept
{
    return data_source_finitude_type::pseudo_infinite;
}

}
