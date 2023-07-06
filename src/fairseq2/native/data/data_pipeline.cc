// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/data_pipeline.h"

#include <algorithm>
#include <exception>
#include <system_error>
#include <utility>

#include <fmt/core.h>

#include "fairseq2/native/data/bucket_by_length_data_source.h"
#include "fairseq2/native/data/bucket_data_source.h"
#include "fairseq2/native/data/filter_data_source.h"
#include "fairseq2/native/data/list_data_source.h"
#include "fairseq2/native/data/map_data_source.h"
#include "fairseq2/native/data/prefetch_data_source.h"
#include "fairseq2/native/data/take_data_source.h"
#include "fairseq2/native/data/round_robin_data_source.h"
#include "fairseq2/native/data/shard_data_source.h"
#include "fairseq2/native/data/shuffle_data_source.h"
#include "fairseq2/native/data/skip_data_source.h"
#include "fairseq2/native/data/tape.h"
#include "fairseq2/native/data/yield_from_data_source.h"
#include "fairseq2/native/data/zip_data_source.h"
#include "fairseq2/native/data/zip_file_data_source.h"
#include "fairseq2/native/data/detail/file_system.h"

using namespace fairseq2::detail;

namespace fairseq2 {

std::optional<data>
data_pipeline::next()
{
    check_if_broken();

    ensure_initialized();

    if (!source_)
        return std::nullopt;

    try {
        return source_->next();
    } catch (...) {
        is_broken_ = true;

        throw;
    }
}

void
data_pipeline::reset()
{
    check_if_broken();

    if (!source_)
        return;

    try {
        source_->reset();
    } catch (...) {
        is_broken_ = true;

        throw;
    }
}

void
data_pipeline::record_position(tape &t) const
{
    check_if_broken();

    if (is_initialized()) {
        t.record(true);

        if (!source_)
            return;

        try {
            source_->record_position(t);
        } catch (...) {
            is_broken_ = true;

            throw;
        }
    } else
        t.record(false);
}

void
data_pipeline::reload_position(tape &t)
{
    check_if_broken();

    if (t.read<bool>()) {
        ensure_initialized();

        if (!source_)
            return;

        try {
            source_->reload_position(t);
        } catch (...) {
            is_broken_ = true;

            throw;
        }
    } else
        reset();
}

inline bool
data_pipeline::is_initialized() const noexcept
{
    return factory_ == nullptr;
}

void
data_pipeline::ensure_initialized()
{
    if (factory_ == nullptr)
        return;

    data_source_factory factory = std::exchange(factory_, nullptr);

    try {
        source_ = factory();
    } catch (...) {
        is_broken_ = true;

        throw;
    }
}

void
data_pipeline::check_if_broken() const
{
    if (is_broken_)
        throw data_pipeline_error{"The data pipeline is broken by a previous operation."};
}

data_pipeline_builder
data_pipeline_builder::bucket(std::size_t bucket_size, bool drop_remainder) &&
{
    if (bucket_size == 0)
        throw std::invalid_argument{"`bucket_size` must be greater than zero."};

    factory_ = [=, inner = std::move(factory_)]
    {
        return std::make_unique<bucket_data_source>(inner(), bucket_size, drop_remainder);
    };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::bucket_by_length(
    std::vector<std::pair<std::size_t, std::size_t>> bucket_sizes,
    std::size_t max_data_length,
    data_length_fn fn,
    bool drop_remainder,
    bool warn_only) &&
{
    if (bucket_sizes.empty())
        throw std::invalid_argument{"`bucket_sizes` must contain at least one element."};

    std::sort(bucket_sizes.begin(), bucket_sizes.end(), [](auto x, auto y)
    {
        return x.second < y.second;
    });

    std::size_t max_bucket_length = bucket_sizes.back().second;
    if (max_data_length > max_bucket_length)
        throw std::invalid_argument{
            fmt::format("`max_data_length` must be less than or equal to {}, but is {} instead.", max_bucket_length, max_data_length)};

    factory_ = [
        =,
        bucket_sizes = std::move(bucket_sizes),
        fn = std::move(fn),
        inner = std::move(factory_)]() mutable
    {
        return std::make_unique<bucket_by_length_data_source>(
            inner(),
            std::move(bucket_sizes),
            max_data_length,
            std::move(fn),
            drop_remainder,
            warn_only);
    };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::filter(predicate_fn fn) &&
{
    factory_ = [fn = std::move(fn), inner = std::move(factory_)]() mutable
    {
        return std::make_unique<filter_data_source>(inner(), std::move(fn));
    };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::map(map_fn fn, std::size_t num_parallel_calls, bool warn_only) &&
{
    factory_ = [=, fn = std::move(fn), inner = std::move(factory_)]() mutable
    {
        return std::make_unique<map_data_source>(
            inner(), std::move(fn), num_parallel_calls, warn_only);
    };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::prefetch(std::size_t num_examples) &&
{
    if (num_examples > 0)
        factory_ = [=, inner = std::move(factory_)]
        {
            return std::make_unique<prefetch_data_source>(inner(), num_examples);
        };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::shard(std::size_t shard_idx, std::size_t num_shards) &&
{
    if (shard_idx >= num_shards)
        throw std::invalid_argument{
            fmt::format("`shard_idx` must be less than `num_shards` ({}), but is {} instead.", num_shards, shard_idx)};

    if (num_shards > 1)
        factory_ = [=, inner = std::move(factory_)]
        {
            return std::make_unique<shard_data_source>(inner(), shard_idx, num_shards);
        };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::shuffle(std::size_t shuffle_window, bool strict, bool enabled) &&
{
    if (enabled)
        factory_ = [=, inner = std::move(factory_)]
        {
            return std::make_unique<shuffle_data_source>(inner(), shuffle_window, strict);
        };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::skip(std::size_t num_examples) &&
{
    if (num_examples > 0)
        factory_ = [=, inner = std::move(factory_)]
        {
            return std::make_unique<skip_data_source>(inner(), num_examples);
        };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::take(std::size_t num_examples) &&
{
    factory_ = [=, inner = std::move(factory_)]
    {
        return std::make_unique<take_data_source>(inner(), num_examples);
    };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::yield_from(yield_fn fn) &&
{
    factory_ = [fn = std::move(fn), inner = std::move(factory_)]() mutable
    {
        return std::make_unique<yield_from_data_source>(inner(), std::move(fn));
    };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline::zip(
    std::vector<data_pipeline> pipelines,
    std::optional<std::vector<std::string>> names,
    bool warn_only,
    bool disable_parallelism)
{
    if (names)
        if (pipelines.size() != names->size())
            throw std::invalid_argument{
                fmt::format("The number of `pipelines` and the number of `names` must be equal, but are {} and {} instead.", pipelines.size(), names->size())};

    bool is_broken = std::any_of(
        pipelines.begin(), pipelines.end(), [](const data_pipeline &pipeline)
        {
            return pipeline.is_broken();
        });

    if (is_broken)
        throw data_pipeline_error{
            "At least one of the specified data pipelines is broken and cannot be zipped."};

    auto tmp = std::make_shared<std::vector<data_pipeline>>(std::move(pipelines));

    auto factory = [names = std::move(names), tmp, warn_only, disable_parallelism]() mutable
    {
        return std::make_unique<zip_data_source>(
            std::move(*tmp), std::move(names), warn_only, disable_parallelism);
    };

    return data_pipeline_builder{std::move(factory)};
}

data_pipeline_builder
data_pipeline::round_robin(std::vector<data_pipeline> pipelines)
{
    bool is_broken = std::any_of(
        pipelines.begin(), pipelines.end(), [](const data_pipeline &pipeline)
        {
            return pipeline.is_broken();
        });

    if (is_broken)
        throw data_pipeline_error{
            "At least one of the specified data pipelines is broken and cannot be used in round robin."};

    auto tmp = std::make_shared<std::vector<data_pipeline>>(std::move(pipelines));

    auto factory = [tmp]() mutable
    {
        return std::make_unique<round_robin_data_source>(std::move(*tmp));
    };

    return data_pipeline_builder{std::move(factory)};
}

data_pipeline
data_pipeline_builder::and_return() &&
{
    if (factory_ == nullptr)
        throw std::runtime_error{"The data pipeline has already been constructed."};

    data_source_factory factory = std::exchange(factory_, nullptr);

    return data_pipeline{std::move(factory)};
}

void
data_pipeline_error::throw_nested(const std::string &message, std::optional<data> example)
{
    std::throw_with_nested(data_pipeline_error{
        fmt::format("{} See nested exception for details.", message), std::move(example)});
}

data_pipeline_error::~data_pipeline_error() = default;

data_pipeline_builder
list_files(std::string pathname, std::optional<std::string> pattern)
{
    auto factory = [pathname = std::move(pathname), pattern = std::move(pattern)]
    {
        std::vector<data> list{};

        try {
            list = detail::list_files(pathname, pattern);
        } catch (const std::system_error &) {
            data_pipeline_error::throw_nested(
                fmt::format("The list of files under '{}' cannot be retrieved.", pathname));
        }

        return std::make_unique<list_data_source>(std::move(list));
    };

    return data_pipeline_builder{std::move(factory)};
}

data_pipeline_builder
read_list(std::vector<data> list)
{
    auto factory = [list = std::move(list)]() mutable
    {
        return std::make_unique<list_data_source>(std::move(list));
    };

    return data_pipeline_builder{std::move(factory)};
}

data_pipeline_builder
read_zipped_records(std::string pathname)
{
    auto factory = [pathname = std::move(pathname)]() mutable
    {
        return std::make_unique<zip_file_data_source>(std::move(pathname));
    };

    return data_pipeline_builder{std::move(factory)};
}

}  // namespace fairseq2
