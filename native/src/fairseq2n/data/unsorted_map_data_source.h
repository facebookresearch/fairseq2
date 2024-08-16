// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class unsorted_map_data_source final : public data_source {
    enum class unsorted_map_state { not_running, running, eod, faulted };

public:
    explicit
    unsorted_map_data_source(
        std::unique_ptr<data_source> &&inner, 
        std::vector<map_fn> &&fns,
        std::size_t buffer_size, 
        std::size_t num_threads) noexcept
      : inner_{std::move(inner)}, 
        map_fns_{std::move(fns)},
        buffer_size_{buffer_size}, 
        prefetch_threads_(num_threads)
    {}

    unsorted_map_data_source(const unsorted_map_data_source &) = delete;
    unsorted_map_data_source & operator=(const unsorted_map_data_source &) = delete;

    unsorted_map_data_source(unsorted_map_data_source &&) = delete;
    unsorted_map_data_source & operator=(unsorted_map_data_source &&) = delete;

   ~unsorted_map_data_source() override;

    std::optional<data>
    next() override;

    void
    reset(bool reset_rng) override;

    void
    record_position(tape &t, bool strict) const override;

    void
    reload_position(tape &t, bool strict) override;

    data_source_finitude_type
    finitude_type() const noexcept override;

private:
    void
    ensure_prefetch_thread_running();

    void
    prefetch(std::size_t thread_idx);

    void
    stop_prefetch_threads() const noexcept;

    void
    join_all_threads() const noexcept;

private:
    std::unique_ptr<data_source> inner_;
    std::vector<map_fn> map_fns_;
    std::size_t buffer_size_;
    unsorted_map_state state_ = unsorted_map_state::not_running;
    mutable std::vector<std::thread> prefetch_threads_{};
    mutable bool should_stop_prefetch_ = false;
    mutable std::mutex queue_mutex_{};
    mutable std::mutex pipeline_mutex_{};
    mutable std::condition_variable fill_queue_condition_{};
    mutable std::condition_variable read_queue_condition_{};
    std::deque<data> fill_queue_{};
    std::deque<data> next_queue_{};
    std::exception_ptr exception_ptr_{};
};

}  // namespace fairseq2n::detail
