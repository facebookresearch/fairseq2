// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/concat_data_source.h"

namespace fairseq2n::detail {

std::optional<data>
concat_data_source::next()
{
    for (data_pipeline &p : pipelines_) {
        if (std::optional<data> maybe_example = p.next())
            return maybe_example;
    }

    return std::nullopt;
}

void concat_data_source::reset()
{
    for (data_pipeline &p : pipelines_)
        p.reset();
}

void concat_data_source::record_position(tape &t) const
{
    for (const data_pipeline &p : pipelines_)
        p.record_position(t);
}

void concat_data_source::reload_position(tape &t)
{
    for (data_pipeline &p : pipelines_)
        p.reload_position(t);
}

} // namespace fairseq2n::detail
