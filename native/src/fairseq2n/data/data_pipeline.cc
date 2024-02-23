// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/data_pipeline.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <system_error>
#include <utility>

#include "data_pipeline.h"
#include "fairseq2n/data/bucket_by_length_data_source.h"
#include "fairseq2n/data/bucket_data_source.h"
#include "fairseq2n/data/concat_data_source.h"
#include "fairseq2n/data/constant_data_source.h"
#include "fairseq2n/data/count_data_source.h"
#include "fairseq2n/data/detail/file_system.h"
#include "fairseq2n/data/filter_data_source.h"
#include "fairseq2n/data/list_data_source.h"
#include "fairseq2n/data/map_data_source.h"
#include "fairseq2n/data/prefetch_data_source.h"
#include "fairseq2n/data/round_robin_data_source.h"
#include "fairseq2n/data/sample_data_source.h"
#include "fairseq2n/data/shard_data_source.h"
#include "fairseq2n/data/shuffle_data_source.h"
#include "fairseq2n/data/skip_data_source.h"
#include "fairseq2n/data/take_data_source.h"
#include "fairseq2n/data/tape.h"
#include "fairseq2n/data/yield_from_data_source.h"
#include "fairseq2n/data/zip_data_source.h"
#include "fairseq2n/data/zip_file_data_source.h"
#include "fairseq2n/detail/exception.h"

using namespace fairseq2n::detail;

namespace fairseq2n {

std::optional<data>
data_pipeline::next()
{
    check_if_broken();

    ensure_initialized();

    if (!source_)
        return std::nullopt;

    while (true) {
        try {
            return source_->next();
        } catch (const data_pipeline_error &ex) {
            if (ex.recoverable() && warning_count_ < max_num_warnings_) {
                warning_count_++;

                // TODO: log exception
            } else {
                if (max_num_warnings_ > 0) {
                    // TODO: log max number of warnings reached.
                }

                // If the error is not recoverable, any further attempt to read
                // from this pipeline will fail immediately.
                if (!ex.recoverable())
                    is_broken_ = true;

                throw;
            }
        } catch (const std::exception &) {
            is_broken_ = true;

            throw;
        }
    }
}

void
data_pipeline::reset()
{
    if (is_broken_ || !source_)
        return;

    try {
        source_->reset();
    } catch (const std::exception &) {
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
        } catch (const std::exception &) {
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
        } catch (const std::exception &) {
            is_broken_ = true;

            throw;
        }
    } else
        reset();
}

bool
data_pipeline::is_infinite() const
{
    check_if_broken();

    ensure_initialized();

    if (!source_)
        return false;

    return source_->is_infinite();
}

inline bool
data_pipeline::is_initialized() const noexcept
{
    return factory_ == nullptr;
}

void
data_pipeline::ensure_initialized() const
{
    if (factory_ == nullptr)
        return;

    data_source_factory factory = std::exchange(factory_, nullptr);

    try {
        source_ = factory();
    } catch (const std::exception &) {
        is_broken_ = true;

        throw;
    }
}

void
data_pipeline::check_if_broken() const
{
    if (is_broken_)
        throw_<data_pipeline_error>(
            "The data pipeline is broken by a previous operation and cannot be used.");
}

data_pipeline_builder
data_pipeline::concat(std::vector<data_pipeline> pipelines)
{
    bool is_broken = std::any_of(
        pipelines.begin(), pipelines.end(), [](const data_pipeline &pipeline)
        {
            return pipeline.is_broken();
        });

    if (is_broken)
        throw_<std::invalid_argument>(
            "At least one of the specified data pipelines is broken and cannot be concatenated.");

    auto tmp = std::make_shared<std::vector<data_pipeline>>(std::move(pipelines));

    auto factory = [tmp]() mutable
    {
        return std::make_unique<concat_data_source>(std::move(*tmp));
    };

    return data_pipeline_builder{std::move(factory)};
}

data_pipeline_builder
data_pipeline::constant(data example, std::optional<std::string> key)
{
    auto factory = [example = std::move(example), key = std::move(key)]() mutable
    {
        return std::make_unique<constant_data_source>(std::move(example), std::move(key));
    };

    return data_pipeline_builder{std::move(factory)};
}

data_pipeline_builder
data_pipeline::count(std::int64_t start, std::int64_t step, std::optional<std::string> key)
{
    auto factory = [start, step, key = std::move(key)]() mutable
    {
        return std::make_unique<count_data_source>(start, step, std::move(key));
    };

    return data_pipeline_builder{std::move(factory)};
}

data_pipeline_builder
data_pipeline::round_robin(std::vector<data_pipeline> pipelines, bool stop_at_shortest)
{
    bool is_broken = std::any_of(
        pipelines.begin(), pipelines.end(), [](const data_pipeline &pipeline)
        {
            return pipeline.is_broken();
        });

    if (is_broken)
        throw_<std::invalid_argument>(
            "At least one of the specified data pipelines is broken and cannot be used in round robin.");

    auto tmp = std::make_shared<std::vector<data_pipeline>>(std::move(pipelines));

    auto factory = [tmp, stop_at_shortest]() mutable
    {
        return std::make_unique<round_robin_data_source>(std::move(*tmp), stop_at_shortest);
    };

    return data_pipeline_builder{std::move(factory)};
}

data_pipeline_builder
data_pipeline::sample(
    std::vector<data_pipeline> pipelines, std::optional<std::vector<float32>> maybe_weights)
{
    bool is_broken = std::any_of(
        pipelines.begin(), pipelines.end(), [](const data_pipeline &pipeline)
        {
            return pipeline.is_broken();
        });

    if (is_broken)
        throw_<std::invalid_argument>(
            "At least one of the specified data pipelines is broken and cannot be sampled.");

    std::vector<float32> weights{};

    if (maybe_weights)
        weights = *maybe_weights;
    else if (!pipelines.empty())
        weights = std::vector<float32>(
            pipelines.size(), 1.0F / static_cast<float32>(pipelines.size()));

    if (weights.size() != pipelines.size())
        throw_<std::invalid_argument>(
            "The number of `pipelines` and the number of `weights` must be equal, but are {} and {} instead.", pipelines.size(), weights.size());

    for (std::size_t i = 0; i < weights.size(); i++) {
        float32 weight = weights[i];

        if (weight < 0.0F || are_close(weight, 0.0F))
            throw_<std::invalid_argument>(
                "The `weights` must be greater than 0.0, but the weight at index {} is {} instead.", i, weight);

        if (!std::isfinite(weight))
            throw_<std::invalid_argument>(
                "The `weights` must be finite, but the weight at index {} is infinite or NaN instead.", i);
    }

    auto tmp = std::make_shared<std::vector<data_pipeline>>(std::move(pipelines));

    auto factory = [tmp, weights=std::move(weights)]() mutable {
        return std::make_unique<sample_data_source>(std::move(*tmp), std::move(weights));
    };

    return data_pipeline_builder{std::move(factory)};
}

data_pipeline_builder
data_pipeline::zip(
    std::vector<data_pipeline> pipelines,
    std::vector<std::string> names,
    bool zip_to_shortest,
    bool flatten,
    bool disable_parallelism)
{
    if (!names.empty() && flatten)
        throw_<std::invalid_argument>(
            "`names` and `flatten` are mutually exclusive and cannot be specified at the same time.");

    if (!names.empty() && pipelines.size() != names.size())
        throw_<std::invalid_argument>(
            "The number of `pipelines` and the number of `names` must be equal, but are {} and {} instead.", pipelines.size(), names.size());

    bool is_broken = std::any_of(
        pipelines.begin(), pipelines.end(), [](const data_pipeline &pipeline)
        {
            return pipeline.is_broken();
        });

    if (is_broken)
        throw_<std::invalid_argument>(
            "At least one of the specified data pipelines is broken and cannot be zipped.");

    auto tmp = std::make_shared<std::vector<data_pipeline>>(std::move(pipelines));

    auto factory = [
        names = std::move(names), tmp, zip_to_shortest, flatten, disable_parallelism]() mutable
    {
        return std::make_unique<zip_data_source>(
            std::move(*tmp), std::move(names), zip_to_shortest, flatten, disable_parallelism);
    };

    return data_pipeline_builder{std::move(factory)};
}

data_pipeline_builder
data_pipeline_builder::bucket(std::size_t bucket_size, bool drop_remainder) &&
{
    if (bucket_size == 0)
        throw_<std::invalid_argument>("`bucket_size` must be greater than zero.");

    factory_ = [=, inner = std::move(factory_)]
    {
        return std::make_unique<bucket_data_source>(inner(), bucket_size, drop_remainder);
    };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::bucket_by_length(
    std::vector<std::pair<std::size_t, std::size_t>> bucket_sizes,
    data_length_fn fn,
    bool skip_long_examples,
    bool drop_remainder) &&
{
    if (bucket_sizes.empty())
        throw_<std::invalid_argument>("`bucket_sizes` must contain at least one element.");

    std::sort(
        bucket_sizes.begin(), bucket_sizes.end(), [](auto x, auto y)
        {
            return x.second < y.second;
        });

    factory_ = [
        =,
        bucket_sizes = std::move(bucket_sizes),
        fn = std::move(fn),
        inner = std::move(factory_)]() mutable
    {
        return std::make_unique<bucket_by_length_data_source>(
            inner(), std::move(bucket_sizes), std::move(fn), skip_long_examples, drop_remainder);
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
data_pipeline_builder::map(map_fn fn, std::size_t num_parallel_calls) &&
{
    factory_ = [=, fn = std::move(fn), inner = std::move(factory_)]() mutable
    {
        return std::make_unique<map_data_source>(inner(), std::move(fn), num_parallel_calls);
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
        throw_<std::invalid_argument>(
            "`shard_idx` must be less than `num_shards` ({}), but is {} instead.", num_shards, shard_idx);

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

data_pipeline
data_pipeline_builder::and_return(std::size_t max_num_warnings) &&
{
    if (factory_ == nullptr)
        throw_<std::domain_error>("The data pipeline has already been constructed.");

    data_source_factory factory = std::exchange(factory_, nullptr);

    return data_pipeline{std::move(factory), max_num_warnings};
}

data_pipeline_error::~data_pipeline_error() = default;

data_pipeline_builder
list_files(std::string pathname, std::optional<std::string> maybe_pattern)
{
    auto factory = [pathname = std::move(pathname), maybe_pattern = std::move(maybe_pattern)]
    {
        data_list list{};

        try {
            list = detail::list_files(pathname, maybe_pattern);
        } catch (const std::system_error &) {
            throw_with_nested<data_pipeline_error>(
                "The list of files under '{}' cannot be retrieved.", pathname);
        }

        return std::make_unique<list_data_source>(std::move(list));
    };

    return data_pipeline_builder{std::move(factory)};
}

data_pipeline_builder
read_list(data_list list)
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

}  // namespace fairseq2n
