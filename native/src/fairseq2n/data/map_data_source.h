// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <semaphore>
#include <utility>
#include <vector>

#include "fairseq2n/detail/thread_pool.h"
#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class map_data_source final : public data_source {
public:
    explicit
    map_data_source(
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

    std::size_t
    acquire_all_available(std::counting_semaphore<>& sem);

private:
    std::unique_ptr<data_source> inner_;
    std::vector<map_fn> map_fns_;
    std::size_t num_parallel_calls_;
    bool deterministic_;
    std::vector<std::optional<data>> buffer_{};
    std::vector<std::optional<data>>::iterator buffer_pos_{};

    mutable std::mutex async_output_queue_mutex_{};
    std::queue<std::optional<data>> async_output_queue_{};
    mutable std::condition_variable read_output_condition_{};
    mutable std::counting_semaphore<> available_workers_semaphore_;
    thread_pool pool_;

};

}  // namespace fairseq2n::detail
