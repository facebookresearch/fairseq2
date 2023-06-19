// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/zipped_data_source.h"

#include <algorithm>

#include <oneapi/tbb.h>

#include "fairseq2/native/data/data_pipeline.h"

namespace fairseq2::detail {

std::optional<data>
zipped_data_source::next()
{
    if (pipelines_.empty())
        return std::nullopt;

    std::vector<data> zip(pipelines_.size());
    std::vector<bool> eod(pipelines_.size());

    // Fetch the next set of elements from the zipped data pipelines.
    auto fn = [this, &zip, &eod](const tbb::blocked_range<std::size_t> &rng) {
        for (auto i = rng.begin(); i < rng.end(); ++i) {
            std::optional<data> d = pipelines_[i].next();
            if (d)
                zip[i] = *std::move(d);
            else
                eod[i] = true;
        }
    };

    tbb::blocked_range<std::size_t> full_rng{0, pipelines_.size()};

    if (disable_parallelism_ || pipelines_.size() == 1)
        fn(full_rng);
    else
        tbb::parallel_for(full_rng, fn);

    // Check whether all data pipelines are in sync.
    if (std::all_of(eod.begin() + 1, eod.end(), [&eod](bool b) { return b == eod[0]; })) {
        if (eod[0])
            return std::nullopt;

        return zip;
    }

    if (!warn_only_)
        throw data_pipeline_error{
            "The zipped data pipelines are expected to have equal length, but at least one data pipeline has more examples than the others."};

    // TODO: print warning.

    return std::nullopt;
}

void
zipped_data_source::reset()
{
    for (auto &dp : pipelines_)
        dp.reset();
}

void
zipped_data_source::record_position(tape &t) const
{
    for (auto &dp : pipelines_)
        dp.record_position(t);
}

void
zipped_data_source::reload_position(tape &t)
{
    for (auto &dp : pipelines_)
        dp.reload_position(t);
}

}  // namespace fairseq2::detail
