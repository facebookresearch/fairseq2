// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <atomic>

#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class unsorted_map_data_source final : public data_source {
public:
    explicit
    unsorted_map_data_source(
        std::unique_ptr<data_source> &&inner,
        std::vector<map_fn> &&fns,
        std::size_t num_parallel_calls) 
    : inner_{std::move(inner)},
      map_fns_{std::move(fns)},
      num_parallel_calls_{num_parallel_calls}
    {
        for (std::size_t i = 0; i < num_parallel_calls; ++i) {
            buffer_.emplace_back(std::nullopt);
        }
    }

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
    bool
    fill_buffer();

    std::optional<data>
    invoke_function(data &&example, std::size_t fn_idx);

    void
    run_map();

    void
    ensure_thread_pool_running();

    void
    stop_thread_pool();

private:
    std::unique_ptr<data_source> inner_;
    std::vector<map_fn> map_fns_;
    std::size_t num_parallel_calls_;
    std::vector<std::thread> thread_pool_;
    std::vector<std::atomic<std::optional<data>>> buffer_{};

    bool faulted_{false};
    bool should_stop_map_{false};

    mutable std::mutex task_queue_mutex_{};
    mutable std::condition_variable task_queue_condition_{};
    std::deque<std::vector<std::atomic<std::optional<data>>>::iterator> task_queue_;

    mutable std::mutex result_queue_mutex_{};
    mutable std::condition_variable result_queue_condition_{};
    std::deque<std::variant<data, std::exception_ptr>> result_queue_{};

    std::exception_ptr exception_ptr_{};
};

}  // namespace fairseq2n::detail
