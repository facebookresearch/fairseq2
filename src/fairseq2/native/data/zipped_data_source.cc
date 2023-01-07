// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/zipped_data_source.h"

#include <algorithm>
#include <limits>

#include <oneapi/tbb.h>

namespace fairseq2::detail {

std::optional<data>
zipped_data_source::next()
{
    std::vector<data> zip(data_pipelines_.size());

    bool eod = false;

    auto fn = [this, &zip, &eod](const tbb::blocked_range<std::size_t> &rng) {
        for (auto i = rng.begin(); i < rng.end(); ++i) {
            std::optional<data> d = data_pipelines_[i].next();
            if (!d) {
                eod = true;

                break;
            }

            zip[i] = *std::move(d);
        }
    };

    parallel_for(tbb::blocked_range<std::size_t>(0, data_pipelines_.size()), fn);

    if (eod)
        return {};

    return zip;
}

std::size_t
zipped_data_source::skip(std::size_t num_examples)
{
    auto n = std::numeric_limits<std::size_t>::max();

    for (auto &dp : data_pipelines_)
        n = std::min(n, dp.skip(num_examples));

    return n;
}

void
zipped_data_source::reset()
{
    for (auto &dp : data_pipelines_)
        dp.reset();
}

void
zipped_data_source::record_position(tape &t) const
{
    for (auto &dp : data_pipelines_)
        dp.record_position(t);
}

void
zipped_data_source::reload_position(tape &t)
{
    for (auto &dp : data_pipelines_)
        dp.reload_position(t);
}

}  // namespace fairseq2::detail
