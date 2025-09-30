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

#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/data_source.h"
#include "fairseq2n/detail/thread_pool.h"

#ifdef FAIRSEQ2N_USE_TBB
#include <oneapi/tbb.h>
#endif

namespace fairseq2n::detail {

class map_data_source final : public data_source {
public:
    explicit map_data_source(
        std::unique_ptr<data_source> &&inner,
        std::vector<map_fn> &&fns,
        std::size_t num_parallel_calls,
        bool deterministic_);

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

    bool
    fill_buffer_async();

    bool
    has_async_output();

    void
    wait_until_done() const;

    void
    reset_async_state();

    struct AsyncMapTask {
        map_data_source *p_this;
        mutable data example;  // 'mutable' so we can move from it in a const call operator

    public:
        void
        operator()() const;
    };

private:
    mutable std::mutex async_input_mutex_{};
    std::unique_ptr<data_source> inner_;
    std::vector<map_fn> map_fns_;
    std::size_t num_parallel_calls_;
    bool deterministic_;
    std::vector<std::optional<data>> buffer_{};
    std::vector<std::optional<data>>::iterator buffer_pos_{};

    mutable std::mutex async_output_mutex_{};
    std::deque<std::optional<data>> async_queue_{};
    mutable std::condition_variable read_output_condition_{};
    std::atomic<bool> finished_{false};
    std::atomic<size_t> tasks_in_flight_{0};
    std::exception_ptr exception_ptr_{};
#ifdef FAIRSEQ2N_USE_TBB
    tbb::task_arena pool_;
#else
    mutable thread_pool pool_;
#endif
};

}  // namespace fairseq2n::detail
