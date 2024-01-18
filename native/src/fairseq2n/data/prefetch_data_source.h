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

#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class prefetch_data_source final : public data_source {
    enum class prefetch_state { not_running, running, eod, faulted };

public:
    explicit
    prefetch_data_source(std::unique_ptr<data_source> &&inner, std::size_t num_examples) noexcept
      : inner_{std::move(inner)}, num_examples_{num_examples}
    {}

    prefetch_data_source(const prefetch_data_source &) = delete;
    prefetch_data_source & operator=(const prefetch_data_source &) = delete;

    prefetch_data_source(prefetch_data_source &&) = delete;
    prefetch_data_source & operator=(prefetch_data_source &&) = delete;

   ~prefetch_data_source() override;

    std::optional<data>
    next() override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

    bool
    is_infinite() const noexcept override;

private:
    void
    ensure_prefetch_thread_running();

    void
    prefetch();

    void
    stop_prefetch_thread() const noexcept;

private:
    std::unique_ptr<data_source> inner_;
    std::size_t num_examples_;
    prefetch_state state_ = prefetch_state::not_running;
    mutable std::thread prefetch_thread_{};
    mutable bool should_stop_prefetch_ = false;
    mutable std::mutex queue_mutex_{};
    mutable std::condition_variable fill_queue_condition_{};
    mutable std::condition_variable read_queue_condition_{};
    std::deque<data> fill_queue_{};
    std::deque<data> next_queue_{};
    std::exception_ptr exception_ptr_{};
};

}  // namespace fairseq2n::detail
