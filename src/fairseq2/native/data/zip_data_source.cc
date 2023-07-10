// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/zip_data_source.h"

#include <algorithm>
#include <cstdint>

#include <fmt/core.h>
#include <fmt/format.h>
#include <oneapi/tbb.h>

#include "fairseq2/native/data/data_pipeline.h"

namespace fairseq2::detail {

zip_data_source::zip_data_source(
    std::vector<data_pipeline> &&pipelines,
    std::optional<std::vector<std::string>> &&names,
    bool flatten,
    bool warn_only,
    bool disable_parallelism) noexcept
  : pipelines_(std::move(pipelines)),
    flatten_{flatten},
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

    data_list zip(pipelines_.size());

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

    if (!are_in_sync) {
        // Collect pipelines that have not reached their end of data yet.
        std::vector<std::size_t> not_eod{};
        for (std::size_t i = 0; i < is_eod.size(); ++i)
            if (is_eod[i] == 0)
                not_eod.push_back(i);

        if (!warn_only_)
            throw data_pipeline_error{
                fmt::format("The zipped data pipelines must all have the same length, but the data pipelines at the following indices have more examples than the others. Indices: {}", fmt::join(not_eod, ", "))};

        // TODO: print warning.

        return std::nullopt;
    }

    if (is_eod[0] == 1)
        return std::nullopt;

    if (flatten_) {
        // We expect all pipelines to match the return type of the first
        // pipeline.
        if (zip[0].is_dict())
            return flatten_to_dict(zip);
        else
            return flatten_to_list(zip);
    }

    // If no names specified, return as list.
    if (names_.empty())
        return zip;

    // Otherwise, as dictionary.
    data_dict dict{};

    for (std::size_t i = 0; i < zip.size(); ++i)
        dict.emplace(names_[i], std::move(zip[i]));

    return dict;
}

std::optional<data>
zip_data_source::flatten_to_dict(data_list &zip) const
{
    data_dict output{};

    for (data &d : zip) {
        // If the first pipeline has returned a `dict`, we expect all other
        // pipelines to return dicts as well.
        if (d.is_dict())
            for (auto &[key, value] : d.as_dict()) {
                // All keys in the flattened dict must be unique.
                if (auto pos = output.find(key); pos != output.end()) {
                    if (!warn_only_)
                        throw data_pipeline_error{
                            fmt::format("The zipped data pipelines must all return unique keys when `flatten` is set, but the key '{}' is not unique.", key)};

                    // TODO: warn

                    return std::nullopt;
                }

                output.emplace(key, std::move(value));
            }
        else {
            if (!warn_only_)
                throw data_pipeline_error{
                    "The zipped data pipelines must all return only dicts, or only non-dicts when `flatten` is set.", std::move(zip)};

            // TODO: warn

            return std::nullopt;
        }
    }

    return output;
}

std::optional<data>
zip_data_source::flatten_to_list(data_list &zip) const
{
    data_list output{};

    for (data &d : zip) {
        // If the first pipeline has returned an example that is not `dict`, we
        // expect all other pipelines to return non-dicts as well.
        if (d.is_dict()) {
            if (!warn_only_)
                throw data_pipeline_error{
                    "The zipped data pipelines must all return only dicts, or only non-dicts when `flatten` is set.", std::move(zip)};

            // TODO: warn

            return std::nullopt;
        }

        if (d.is_list())
            for (data &element : d.as_list())
                output.push_back(std::move(element));
        else
            output.push_back(std::move(d));
    }

    return output;
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
