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

#include "fairseq2/native/py.h"
#include "fairseq2/native/data/bucket_by_length_data_source.h"
#include "fairseq2/native/data/bucket_data_source.h"
#include "fairseq2/native/data/data_processor.h"
#include "fairseq2/native/data/filtered_data_source.h"
#include "fairseq2/native/data/list_data_source.h"
#include "fairseq2/native/data/mapped_data_source.h"
#include "fairseq2/native/data/prefetched_data_source.h"
#include "fairseq2/native/data/ranged_data_source.h"
#include "fairseq2/native/data/round_robin_data_source.h"
#include "fairseq2/native/data/sharded_data_source.h"
#include "fairseq2/native/data/shuffled_data_source.h"
#include "fairseq2/native/data/skipped_data_source.h"
#include "fairseq2/native/data/tape.h"
#include "fairseq2/native/data/yielded_data_source.h"
#include "fairseq2/native/data/zip_data_source.h"
#include "fairseq2/native/data/zipped_data_source.h"
#include "fairseq2/native/data/detail/file_system.h"
#include "fairseq2/native/data/processors/custom_data_processor.h"

using namespace fairseq2::detail;

namespace fairseq2 {

std::optional<data>
data_pipeline::next()
{
    check_if_broken();

    ensure_initialized();

    if (!src_)
        return {};

    {
        py_gil_release no_gil{};

        try {
            return src_->next();
        } catch (...) {
            is_broken_ = true;

            throw;
        }
    }
}

void
data_pipeline::reset()
{
    check_if_broken();

    if (!src_)
        return;

    {
        py_gil_release no_gil{};

        try {
            src_->reset();
        } catch (...) {
            is_broken_ = true;

            throw;
        }
    }
}

void
data_pipeline::record_position(tape &t) const
{
    check_if_broken();

    if (initialized()) {
        t.record(true);

        if (!src_)
            return;

        {
            py_gil_release no_gil{};

            try {
                src_->record_position(t);
            } catch (...) {
                is_broken_ = true;

                throw;
            }
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

        if (!src_)
            return;

        {
            py_gil_release no_gil{};

            try {
                src_->reload_position(t);
            } catch (...) {
                is_broken_ = true;

                throw;
            }
        }
    } else
        reset();
}

inline bool
data_pipeline::initialized() const noexcept
{
    return factory_ == nullptr;
}

void
data_pipeline::ensure_initialized()
{
    if (factory_ == nullptr)
        return;

    data_source_factory fn = std::exchange(factory_, nullptr);

    try {
        src_ = fn();
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

    factory_ = [=, inner = std::move(factory_)]()
    {
        return std::make_unique<bucket_data_source>(inner(), bucket_size, drop_remainder);
    };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::bucket_by_length(
    std::vector<std::pair<std::size_t, std::size_t>> bucket_sizes,
    std::size_t max_data_length,
    data_length_fn f,
    bool drop_remainder,
    bool warn_only) &&
{
    if (bucket_sizes.empty())
        throw std::invalid_argument{"`bucket_sizes` must contain at least one element."};

    std::sort(bucket_sizes.begin(), bucket_sizes.end(), [](auto x, auto y) {
        return x.second < y.second;
    });

    std::size_t max_bucket_length = bucket_sizes.back().second;
    if (max_data_length > max_bucket_length)
        throw std::invalid_argument{
            fmt::format("`max_data_length` must be less than or equal to {}, but is {} instead.", max_bucket_length, max_data_length)};

    factory_ = [
        =,
        bucket_sizes = std::move(bucket_sizes),
        f = std::move(f),
        inner = std::move(factory_)]() mutable
    {
        return std::make_unique<bucket_by_length_data_source>(
            inner(),
            std::move(bucket_sizes),
            max_data_length,
            std::move(f),
            drop_remainder,
            warn_only);
    };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::filter(predicate_fn f) &&
{
    factory_ = [f = std::move(f), inner = std::move(factory_)]() mutable {
        return std::make_unique<filtered_data_source>(inner(), std::move(f));
    };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::map(map_fn f, std::size_t num_parallel_calls, bool warn_only) &&
{
    return std::move(*this).map(
        std::make_shared<custom_data_processor>(std::move(f)), num_parallel_calls, warn_only);
}

data_pipeline_builder
data_pipeline_builder::map(
    std::shared_ptr<const data_processor> p, std::size_t num_parallel_calls, bool warn_only) &&
{
    factory_ = [=, p = std::move(p), inner = std::move(factory_)]() mutable {
        return std::make_unique<mapped_data_source>(
            inner(), std::move(p), num_parallel_calls, warn_only);
    };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::prefetch(std::size_t num_examples) &&
{
    if (num_examples > 0)
        factory_ = [=, inner = std::move(factory_)]() {
            return std::make_unique<prefetched_data_source>(inner(), num_examples);
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
        factory_ = [=, inner = std::move(factory_)]() {
            return std::make_unique<sharded_data_source>(inner(), shard_idx, num_shards);
        };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::shuffle(std::size_t shuffle_window, bool strict, bool enabled) &&
{
    if (enabled)
        factory_ = [=, inner = std::move(factory_)]() {
            return std::make_unique<shuffled_data_source>(inner(), shuffle_window, strict);
        };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::skip(std::size_t num_examples) &&
{
    if (num_examples > 0)
        factory_ = [=, inner = std::move(factory_)]() {
            return std::make_unique<skipped_data_source>(inner(), num_examples);
        };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::take(std::size_t num_examples) &&
{
    factory_ = [=, inner = std::move(factory_)]() {
        return std::make_unique<ranged_data_source>(inner(), num_examples);
    };

    return std::move(*this);
}

data_pipeline_builder
data_pipeline_builder::yield_from(yield_fn f) &&
{
    factory_ = [f = std::move(f), inner = std::move(factory_)]() mutable {
        return std::make_unique<yielded_data_source>(inner(), std::move(f));
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
    if (names) {
        if (pipelines.size() != names->size())
            throw std::invalid_argument{
                fmt::format("The number of `pipelines` and the number of `names` must be equal, but are {} and {} instead.", pipelines.size(), names->size())};
    }

    bool is_broken = std::any_of(pipelines.begin(), pipelines.end(), [](const data_pipeline &p) {
        return p.is_broken();
    });

    if (is_broken)
        throw data_pipeline_error{
            "At least one of the specified data pipelines is broken and cannot be zipped."};

    auto tmp = std::make_shared<std::vector<data_pipeline>>(std::move(pipelines));

    auto f = [n = std::move(names), tmp, warn_only, disable_parallelism]() mutable {
        return std::make_unique<zipped_data_source>(
            std::move(*tmp), std::move(n), warn_only, disable_parallelism);
    };

    return data_pipeline_builder{std::move(f)};
}

data_pipeline_builder
data_pipeline::round_robin(std::vector<data_pipeline> pipelines)
{
    bool is_broken = std::any_of(pipelines.begin(), pipelines.end(), [](const data_pipeline &p) {
        return p.is_broken();
    });

    if (is_broken)
        throw data_pipeline_error{
            "At least one of the specified data pipelines is broken and cannot be used in round robin."};

    auto tmp = std::make_shared<std::vector<data_pipeline>>(std::move(pipelines));

    auto f = [tmp]() mutable {
        return std::make_unique<round_robin_data_source>(std::move(*tmp));
    };

    return data_pipeline_builder{std::move(f)};
}

data_pipeline
data_pipeline_builder::and_return() &&
{
    if (factory_ == nullptr)
        throw std::runtime_error{"The data pipeline has already been constructed."};

    data_source_factory f = std::exchange(factory_, nullptr);

    return data_pipeline{std::move(f)};
}

void
data_pipeline_error::throw_nested(const std::string &msg, std::optional<data> example)
{
    std::throw_with_nested(data_pipeline_error{
        fmt::format("{} See nested exception for details.", msg), std::move(example)});
}

data_pipeline_error::~data_pipeline_error() = default;

data_pipeline_builder
list_files(std::string pathname, std::optional<std::string> pattern)
{
    auto f = [p = std::move(pathname), pattern = std::move(pattern)]() {
        std::vector<data> lst{};

        try {
            py_gil_release no_gil{};

            lst = detail::list_files(p, pattern);
        } catch (const std::system_error &) {
            data_pipeline_error::throw_nested(
                fmt::format("The list of files under '{}' cannot be retrieved.", p));
        }

        return std::make_unique<list_data_source>(std::move(lst));
    };

    return data_pipeline_builder{std::move(f)};
}

data_pipeline_builder
read_list(std::vector<data> lst)
{
    auto f = [l = std::move(lst)]() mutable {
        return std::make_unique<list_data_source>(std::move(l));
    };

    return data_pipeline_builder{std::move(f)};
}

data_pipeline_builder
read_zipped_records(std::string pathname)
{
    auto f = [p = std::move(pathname)]() mutable {
        return std::make_unique<zip_data_source>(std::move(p));
    };

    return data_pipeline_builder{std::move(f)};
}

}  // namespace fairseq2
