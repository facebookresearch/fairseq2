// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/zipped_data_source.h"

#include <algorithm>
#include <cstdint>

#include <oneapi/tbb.h>

#include "fairseq2/native/data/data_pipeline.h"

namespace fairseq2::detail {

zipped_data_source::zipped_data_source(
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
zipped_data_source::next()
{
    if (pipelines_.empty())
        return std::nullopt;

    std::vector<data> zip(pipelines_.size());

    // Do not use `bool` here as, per standard, it is not thread-safe even for
    // distinct elements.
    std::vector<std::int8_t> eod(pipelines_.size());

    // Fetch the next set of elements from the zipped data pipelines.
    auto f = [this, &zip, &eod](const tbb::blocked_range<std::size_t> &r) {
        for (auto i = r.begin(); i < r.end(); ++i) {
            std::optional<data> d = pipelines_[i].next();
            if (d)
                zip[i] = *std::move(d);
            else
                eod[i] = 1;
        }
    };

    tbb::blocked_range<std::size_t> r{0, pipelines_.size()};

    if (disable_parallelism_ || pipelines_.size() == 1)
        f(r);
    else
        tbb::parallel_for(r, f);

    // Check whether all data pipelines are in sync.
    if (std::all_of(eod.begin() + 1, eod.end(), [&eod](std::int8_t b) { return b == eod[0]; })) {
        if (eod[0] == 1)
            return std::nullopt;

        // If no names specified, return as list.
        if (names_.empty())
            return data{std::move(zip)};

        // Otherwise, as dictionary.
        flat_hash_map<std::string, data> m{};

        for (std::size_t i = 0; i < zip.size(); ++i)
            m.emplace(names_[i], std::move(zip[i]));

        return data{std::move(m)};
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
    for (auto &p : pipelines_)
        p.reset();
}

void
zipped_data_source::record_position(tape &t) const
{
    for (auto &p : pipelines_)
        p.record_position(t);
}

void
zipped_data_source::reload_position(tape &t)
{
    for (auto &p : pipelines_)
        p.reload_position(t);
}

}  // namespace fairseq2::detail
