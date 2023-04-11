// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/data_pipeline.h"

#include <algorithm>
#include <exception>
#include <system_error>

#include <fmt/core.h>

#include "fairseq2/native/py.h"
#include "fairseq2/native/data/batched_data_source.h"
#include "fairseq2/native/data/batched_by_length_data_source.h"
#include "fairseq2/native/data/yielded_data_source.h"
#include "fairseq2/native/data/list_data_source.h"
#include "fairseq2/native/data/mapped_data_source.h"
#include "fairseq2/native/data/prefetched_data_source.h"
#include "fairseq2/native/data/sharded_data_source.h"
#include "fairseq2/native/data/tape.h"
#include "fairseq2/native/data/zipped_data_source.h"
#include "fairseq2/native/data/detail/file_system.h"

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

std::size_t
data_pipeline::skip(std::size_t num_examples)
{
    check_if_broken();

    ensure_initialized();

    if (!src_)
        return 0;

    {
        py_gil_release no_gil{};

        try {
            return src_->skip(num_examples);
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

    if (is_initialized()) {
        t.record(true);

        if (!src_)
            return;

        {
            py_gil_release no_gil{};

            src_->record_position(t);
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
data_pipeline::is_initialized() const noexcept
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

data_pipeline_builder &
data_pipeline_builder::batch(std::size_t batch_size, bool drop_remainder) &
{
    factory_ = [=, inner = std::move(factory_)]() {
        return std::make_unique<batched_data_source>(inner(), batch_size, drop_remainder);
    };

    return *this;
}

data_pipeline_builder &&
data_pipeline_builder::batch(std::size_t batch_size, bool drop_remainder) &&
{
    batch(batch_size, drop_remainder);

    return std::move(*this);
}

data_pipeline_builder &
data_pipeline_builder::batch_by_length(const std::vector<std::pair<std::size_t, std::size_t>>& buffer_sizes, std::int32_t pad_idx) &
{
    factory_ = [=, inner = std::move(factory_)]() {
        return std::make_unique<batched_by_length_data_source>(inner(), buffer_sizes, pad_idx);
    };

    return *this;
}

data_pipeline_builder &&
data_pipeline_builder::batch_by_length(const std::vector<std::pair<std::size_t, std::size_t>>& buffer_sizes, std::int32_t pad_idx) &&
{
    batch_by_length(buffer_sizes, pad_idx);
    return std::move(*this);
}

data_pipeline_builder &
data_pipeline_builder::yield_from(yield_fn fn) &
{
    factory_ = [fn = std::move(fn), inner = std::move(factory_)]() mutable {
        return std::make_unique<yielded_data_source>(inner(), std::move(fn));
    };

    return *this;
}

data_pipeline_builder &&
data_pipeline_builder::yield_from(yield_fn fn) &&
{
    yield_from(std::move(fn));

    return std::move(*this);
}

data_pipeline_builder &
data_pipeline_builder::map(map_fn fn) &
{
    factory_ = [fn = std::move(fn), inner = std::move(factory_)]() mutable {
        return std::make_unique<mapped_data_source>(inner(), std::move(fn), 0);
    };

    return *this;
}

data_pipeline_builder &&
data_pipeline_builder::map(map_fn fn) &&
{
    map(std::move(fn));

    return std::move(*this);
}

data_pipeline_builder &
data_pipeline_builder::map(map_fn  fn, std::size_t chunk_size) &
{
    factory_ = [fn = std::move(fn), inner = std::move(factory_), chunk_size]() mutable {
        return std::make_unique<mapped_data_source>(inner(), std::move(fn), chunk_size);
    };

    return *this;
}

data_pipeline_builder &&
data_pipeline_builder::map(map_fn fn, std::size_t chunk_size) &&
{
    map(std::move(fn), chunk_size);

    return std::move(*this);
}

data_pipeline_builder &
data_pipeline_builder::prefetch(std::size_t num_examples) &
{
    if (num_examples > 0) {
        factory_ = [=, inner = std::move(factory_)]() {
            return std::make_unique<prefetched_data_source>(inner(), num_examples);
        };
    }

    return *this;
}

data_pipeline_builder &&
data_pipeline_builder::prefetch(std::size_t num_examples) &&
{
    prefetch(num_examples);

    return std::move(*this);
}

data_pipeline_builder &
data_pipeline_builder::shard(std::size_t shard_idx, std::size_t num_shards) &
{
    factory_ = [=, inner = std::move(factory_)]() {
        return std::make_unique<sharded_data_source>(inner(), shard_idx, num_shards);
    };

    return *this;
}

data_pipeline_builder &&
data_pipeline_builder::shard(std::size_t shard_idx, std::size_t num_shards) &&
{
    shard(shard_idx, num_shards);

    return std::move(*this);
}

data_pipeline
data_pipeline_builder::and_return() &&
{
    if (factory_ == nullptr)
        throw std::runtime_error{"The data pipeline has already been constructed."};

    data_source_factory fc = std::exchange(factory_, nullptr);

    return data_pipeline{std::move(fc)};
}

void
data_pipeline_error::throw_nested(const std::string &msg, std::optional<data> example)
{
    std::throw_with_nested(data_pipeline_error{
        fmt::format("{} See nested exception for details.", msg), std::move(example)});
}

data_pipeline_error::~data_pipeline_error() = default;

data_pipeline_builder
read_list(std::vector<data> lst)
{
    auto fc = [lst = std::move(lst)]() mutable {
        return std::make_unique<list_data_source>(std::move(lst));
    };

    return data_pipeline_builder{std::move(fc)};
}

data_pipeline_builder
zip_data_pipelines(std::vector<data_pipeline> zip)
{
    bool is_broken = std::any_of(zip.begin(), zip.end(), [](const data_pipeline &dp) {
        return dp.is_broken();
    });

    if (is_broken)
        throw data_pipeline_error{
            "At least one of the specified data pipelines is broken and cannot be zipped."};

    auto sh = std::make_shared<std::vector<data_pipeline>>(std::move(zip));

    auto fc = [sh]() mutable {
        return std::make_unique<zipped_data_source>(std::move(*sh));
    };

    return data_pipeline_builder{std::move(fc)};
}

data_pipeline_builder
list_files(std::string pathname, std::optional<std::string> pattern)
{
    auto fc = [pathname = std::move(pathname), pattern = std::move(pattern)]() {
        std::vector<data> data;

        try {
            py_gil_release no_gil{};

            data = detail::list_files(pathname, pattern);
        } catch (const std::system_error &) {
            data_pipeline_error::throw_nested(
                fmt::format("The list of files under '{}' cannot be retrieved.", pathname));
        }

        return std::make_unique<list_data_source>(std::move(data));
    };

    return data_pipeline_builder{std::move(fc)};
}

}  // namespace fairseq2
