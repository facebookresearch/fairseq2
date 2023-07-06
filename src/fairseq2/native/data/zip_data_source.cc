// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/zip_data_source.h"

#include <algorithm>
#include <cstdint>

#include <oneapi/tbb.h>

#include "fairseq2/native/data/data_pipeline.h"

namespace fairseq2::detail {

zip_data_source::zip_data_source(
    std::vector<data_pipeline> &&pipelines,
    std::optional<std::vector<std::string>> &&names,
    bool warn_only,
    bool disable_parallelism) noexcept
  : pipelines_(std::move(pipelines)),
    warn_only_{warn_only},
    disable_parallelism_{disable_parallelism}
{
    if (names)
        names_ = *std::move(names);
}

std::optional<data>
zip_data_source::next()
{
    if (pipelines_.empty())
        return std::nullopt;

    std::vector<data> zip(pipelines_.size());

    // Do not use `bool` here as, per standard, it is not thread-safe even for
    // distinct elements.
    std::vector<std::int8_t> is_eod(pipelines_.size());

    // Fetch the next set of elements from the zip data pipelines.
    auto fetch_next = [this, &zip, &is_eod](const tbb::blocked_range<std::size_t> &range)
    {
        for (auto i = range.begin(); i < range.end(); ++i) {
            std::optional<data> d = pipelines_[i].next();
            if (d)
                zip[i] = *std::move(d);
            else
                is_eod[i] = 1;
        }
    };

    tbb::blocked_range<std::size_t> range{0, pipelines_.size()};

    if (disable_parallelism_ || pipelines_.size() == 1)
        fetch_next(range);
    else
        tbb::parallel_for(range, fetch_next);

    // Check whether all data pipelines are in sync.
    bool are_in_sync = std::all_of(
        is_eod.begin() + 1, is_eod.end(), [&is_eod](std::int8_t b) { return b == is_eod[0]; });

    if (are_in_sync) {
        if (is_eod[0] == 1)
            return std::nullopt;

        // If no names specified, return as list.
        if (names_.empty())
            return zip;

        // Otherwise, as dictionary.
        flat_hash_map<std::string, data> dict{};

        for (std::size_t i = 0; i < zip.size(); ++i)
            dict.emplace(names_[i], std::move(zip[i]));

        return dict;
    }

    if (!warn_only_)
        throw data_pipeline_error{
            "The zipped data pipelines are expected to have equal length, but at least one data pipeline has more examples than the others."};

    // TODO: print warning.

    return std::nullopt;
}

void
zip_data_source::reset()
{
    for (auto &pipeline : pipelines_)
        pipeline.reset();
}

void
zip_data_source::record_position(tape &t) const
{
    for (auto &pipeline : pipelines_)
        pipeline.record_position(t);
}

void
zip_data_source::reload_position(tape &t)
{
    for (auto &pipeline : pipelines_)
        pipeline.reload_position(t);
}

}  // namespace fairseq2::detail
