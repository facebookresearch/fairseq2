// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data.h"
#include "fairseq2/native/data/data_source.h"
#include "fairseq2/native/data/tape.h"

namespace fairseq2 {

using data_source_factory = std::function<std::unique_ptr<data_source>()>;

class FAIRSEQ2_API data_pipeline {
    friend class data_pipeline_builder;

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
    is_broken() const noexcept
    {
        return is_broken_;
    }

private:
    explicit
    data_pipeline(data_source_factory &&fc) noexcept
        : factory_{std::move(fc)}
    {}

    bool
    is_initialized() const noexcept;

    void
    ensure_initialized();

    void
    check_if_broken() const;

    [[noreturn]] static void
    throw_broken();

private:
    data_source_factory factory_{};
    std::unique_ptr<data_source> src_{};
    bool is_broken_ = false;
};

using map_fn = std::function<data(data &&)>;
using predicate_fn = std::function<bool(const data &)>;
using yield_fn = std::function<data_pipeline(const data &)>;

class FAIRSEQ2_API data_pipeline_builder {
public:
    explicit
    data_pipeline_builder(data_source_factory fc) noexcept
        : factory_{std::move(fc)}
    {}

    data_pipeline_builder(const data_pipeline_builder &) = delete;
    data_pipeline_builder &operator=(const data_pipeline_builder &) = delete;

    data_pipeline_builder(data_pipeline_builder &&) noexcept = default;
    data_pipeline_builder &operator=(data_pipeline_builder &&) noexcept = default;

   ~data_pipeline_builder() = default;

    data_pipeline_builder &&
    batch(std::size_t batch_size, bool drop_remainder = false, const std::vector<std::int32_t> &pad_idx = {}) &&;

    data_pipeline_builder &&
    batch_by_length(const std::vector<std::pair<std::size_t, std::size_t>>& buffer_sizes, std::int32_t pad_idx) &&;

    data_pipeline_builder &&
    yield_from(yield_fn fn) &&;

    data_pipeline_builder &&
    map(map_fn fn) &&;

    data_pipeline_builder &&
    map(map_fn fn, std::size_t chunk_size) &&;

    data_pipeline_builder &&
    filter(predicate_fn predicate) &&;

    data_pipeline_builder &&
    prefetch(std::size_t num_examples) &&;

    data_pipeline_builder &&
    shard(std::size_t shard_idx, std::size_t num_shards) &&;

    data_pipeline_builder &&
    shuffle(std::size_t buffer_size, std::size_t seed, bool deterministic) &&;

    data_pipeline_builder &&
    skip(std::size_t count) &&;

    data_pipeline_builder &&
    take(std::size_t count) &&;

    data_pipeline
    and_return() &&;

private:
    data_source_factory factory_;
};

class FAIRSEQ2_API data_pipeline_error : public std::runtime_error {
public:
    [[noreturn]] static void
    throw_nested(const std::string &msg, std::optional<data> example = {});

public:
    explicit
    data_pipeline_error(const std::string &msg, std::optional<data> &&example = {}) noexcept
        : std::runtime_error{msg}, example_{std::move(example)}
    {}

    data_pipeline_error(const data_pipeline_error &) = default;
    data_pipeline_error &operator=(const data_pipeline_error &) = default;

   ~data_pipeline_error() override;

    const std::optional<data> &
    example() const noexcept
    {
        return example_;
    }

private:
    std::optional<data> example_;
};

FAIRSEQ2_API data_pipeline_builder
read_list(std::vector<data> lst);

FAIRSEQ2_API data_pipeline_builder
zip_data_pipelines(std::vector<data_pipeline> zip);

FAIRSEQ2_API data_pipeline_builder
round_robin_data_pipelines(std::vector<data_pipeline> pipelines, std::vector<float> probs = {});

FAIRSEQ2_API data_pipeline_builder
list_files(std::string pathname, std::optional<std::string> pattern = {});

}  // namespace fairseq2
