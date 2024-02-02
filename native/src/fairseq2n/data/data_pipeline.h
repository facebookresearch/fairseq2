// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fairseq2n/api.h"
#include "fairseq2n/data/data.h"
#include "fairseq2n/data/data_source.h"
#include "fairseq2n/data/tape.h"

namespace fairseq2n {

using data_source_factory = std::function<std::unique_ptr<data_source>()>;

class data_pipeline_builder;

class FAIRSEQ2_API data_pipeline {
    friend class data_pipeline_builder;

private:
    explicit
    data_pipeline(data_source_factory &&factory, std::size_t max_num_warnings) noexcept
      : factory_{std::move(factory)}, max_num_warnings_{max_num_warnings}
    {}

public:
    data_pipeline() noexcept = default;

    std::optional<data>
    next();

    void
    reset();

    void
    record_position(tape &t) const;

    void
    reload_position(tape &t);

    bool
    is_infinite() const;

    bool
    is_broken() const noexcept
    {
        return is_broken_;
    }

private:
    bool
    is_initialized() const noexcept;

    void
    ensure_initialized() const;

    void
    check_if_broken() const;

public:
    static data_pipeline_builder
    concat(std::vector<data_pipeline> pipelines);

    static data_pipeline_builder
    constant(data example, std::optional<std::string> key = {});

    static data_pipeline_builder
    count(std::int64_t start = 0, std::int64_t step = 1, std::optional<std::string> key = {});

    static data_pipeline_builder
    round_robin(std::vector<data_pipeline> pipelines, bool stop_at_shortest = false);

    static data_pipeline_builder
    sample(std::vector<data_pipeline> pipelines, std::optional<std::vector<float>> weights = {});

    static data_pipeline_builder
    zip(
        std::vector<data_pipeline> pipelines,
        std::vector<std::string> names = {},
        bool zip_to_shortest = false,
        bool flatten = false,
        bool disable_parallelism = false);

private:
    mutable data_source_factory factory_{};
    mutable std::unique_ptr<data_source> source_{};
    std::size_t max_num_warnings_{};
    std::size_t warning_count_{};
    mutable bool is_broken_ = false;
};

using data_length_fn = std::function<std::size_t(const data &)>;

using map_fn = std::function<data(data &&)>;

using predicate_fn = std::function<bool(const data &)>;

using yield_fn = std::function<data_pipeline(const data &)>;

class FAIRSEQ2_API data_pipeline_builder {
public:
    explicit
    data_pipeline_builder(data_source_factory factory) noexcept
      : factory_{std::move(factory)}
    {}

    data_pipeline_builder(const data_pipeline_builder &) = delete;
    data_pipeline_builder &operator=(const data_pipeline_builder &) = delete;

    data_pipeline_builder(data_pipeline_builder &&) noexcept = default;
    data_pipeline_builder &operator=(data_pipeline_builder &&) noexcept = default;

   ~data_pipeline_builder() = default;

    data_pipeline_builder
    bucket(std::size_t bucket_size, bool drop_remainder = false) &&;

    data_pipeline_builder
    bucket_by_length(
        std::vector<std::pair<std::size_t, std::size_t>> bucket_sizes,
        data_length_fn fn,
        bool skip_long_examples = false,
        bool drop_remainder = false) &&;

    data_pipeline_builder
    filter(predicate_fn fn) &&;

    data_pipeline_builder
    map(map_fn fn, std::size_t num_parallel_calls = 1) &&;

    data_pipeline_builder
    prefetch(std::size_t num_examples) &&;

    data_pipeline_builder
    shard(std::size_t shard_idx, std::size_t num_shards) &&;

    data_pipeline_builder
    shuffle(std::size_t shuffle_window, bool strict, bool enabled = true) &&;

    data_pipeline_builder
    skip(std::size_t num_examples) &&;

    data_pipeline_builder
    take(std::size_t num_examples) &&;

    data_pipeline_builder
    yield_from(yield_fn fn) &&;

    data_pipeline
    and_return(std::size_t max_num_warnings = 0) &&;

private:
    data_source_factory factory_;
};

class FAIRSEQ2_API data_pipeline_error : public std::runtime_error {
public:
    explicit
    data_pipeline_error(
        const std::string &message,
        std::optional<data> maybe_example = {},
        bool recoverable = false) noexcept
      : std::runtime_error{message},
        maybe_example_{std::move(maybe_example)},
        recoverable_{recoverable}
    {}

    data_pipeline_error(const data_pipeline_error &) = default;
    data_pipeline_error &operator=(const data_pipeline_error &) = default;

   ~data_pipeline_error() override;

    std::optional<data> &
    maybe_example() noexcept
    {
        return maybe_example_;
    }

    const std::optional<data> &
    maybe_example() const noexcept
    {
        return maybe_example_;
    }

    bool
    recoverable() const noexcept
    {
        return recoverable_;
    }

private:
    std::optional<data> maybe_example_{};
    bool recoverable_;
};

FAIRSEQ2_API data_pipeline_builder
list_files(std::string pathname, std::optional<std::string> maybe_pattern = {});

FAIRSEQ2_API data_pipeline_builder
read_list(data_list list);

FAIRSEQ2_API data_pipeline_builder
read_zipped_records(std::string pathname);

}  // namespace fairseq2n
