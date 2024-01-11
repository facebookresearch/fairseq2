// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/zip_data_source.h"

#include <algorithm>
#include <cstdint>

#include <fmt/format.h>

#include "fairseq2n/detail/exception.h"
#include "fairseq2n/detail/parallel.h"
#include "fairseq2n/data/detail/exception.h"

namespace fairseq2n::detail {

zip_data_source::zip_data_source(
    std::vector<data_pipeline> &&pipelines,
    std::vector<std::string> &&names,
    bool zip_to_shortest,
    bool flatten,
    bool disable_parallelism) noexcept
  : pipelines_(std::move(pipelines)),
    names_(std::move(names)),
    zip_to_shortest_{zip_to_shortest},
    flatten_{flatten},
    disable_parallelism_{disable_parallelism}
{}

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
    auto fetch_next = [this, &zip, &is_eod](std::size_t begin, std::size_t end)
    {
        for (auto i = begin; i < end; ++i) {
            std::optional<data> maybe_example = pipelines_[i].next();
            if (maybe_example) {
                zip[i] = *std::move(maybe_example);

                if (pipelines_[i].is_infinite())
                    is_eod[i] = 2;
            } else
                is_eod[i] = 1;
        }
    };

    if (disable_parallelism_ || pipelines_.size() == 1)
        fetch_next(0, pipelines_.size());
    else
        parallel_for<std::size_t>(fetch_next, pipelines_.size());

    // Check whether all data pipelines are in sync.
    bool are_eod = std::all_of(
        is_eod.begin(), is_eod.end(), [](std::int8_t b)
        {
            return b == 2 || b == 1;
        });

    bool are_not_eod = std::all_of(
        is_eod.begin(), is_eod.end(), [](std::int8_t b)
        {
            return b == 2 || b == 0;
        });

    bool are_in_sync = are_eod || are_not_eod;

    if (!are_in_sync) {
        if (zip_to_shortest_)
            return std::nullopt;

        // Collect pipelines that have not reached their end of data yet.
        std::vector<std::size_t> not_eod{};
        for (std::size_t i = 0; i < is_eod.size(); ++i)
            if (is_eod[i] == 0)
                not_eod.push_back(i);

        throw_<data_pipeline_error>(
            "The zipped data pipelines must all have the same number of examples, but the data pipelines at the indices [{}] have more examples than the others.", fmt::join(not_eod, ", "));
    }

    if (are_eod)
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
zip_data_source::flatten_to_dict(data_list &zip)
{
    data_dict output{};

    for (data &example : zip) {
        // If the first pipeline has returned a `dict`, we expect all other
        // pipelines to return dicts as well.
        if (example.is_dict())
            for (auto &[key, value] : example.as_dict()) {
                // All keys in the flattened dict must be unique.
                if (auto pos = output.find(key); pos != output.end())
                    throw_data_pipeline_error(std::nullopt, /*recoverable=*/true,
                        "The zipped data pipelines must all return unique keys when `flatten` is set, but the key '{}' is not unique.", key);

                output.emplace(key, std::move(value));
            }
        else
            throw_data_pipeline_error(std::nullopt, /*recoverable=*/true,
                "The zipped data pipelines must all return only dicts, or only non-dicts when `flatten` is set.");
    }

    return output;
}

std::optional<data>
zip_data_source::flatten_to_list(data_list &zip)
{
    data_list output{};

    for (data &example : zip) {
        // If the first pipeline has returned an example that is not `dict`, we
        // expect all other pipelines to return non-dicts as well.
        if (example.is_dict())
            throw_data_pipeline_error(std::nullopt, /*recoverable=*/true,
                "The zipped data pipelines must all return only dicts, or only non-dicts when `flatten` is set.");

        if (example.is_list())
            for (data &element : example.as_list())
                output.push_back(std::move(element));
        else
            output.push_back(std::move(example));
    }

    return output;
}

void
zip_data_source::reset()
{
    for (data_pipeline &pipeline : pipelines_)
        pipeline.reset();
}

void
zip_data_source::record_position(tape &t) const
{
    for (const data_pipeline &pipeline : pipelines_)
        pipeline.record_position(t);
}

void
zip_data_source::reload_position(tape &t)
{
    for (data_pipeline &pipeline : pipelines_)
        pipeline.reload_position(t);
}

bool
zip_data_source::is_infinite() const noexcept
{
    return false;
}

}  // namespace fairseq2n::detail
