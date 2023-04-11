// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/prefetched_data_source.h"

#include <iostream>
namespace fairseq2::detail {

prefetched_data_source::~prefetched_data_source()
{
    stop_prefetch();
}

std::optional<data>
prefetched_data_source::next()
{
    //               ┌──────────< next() <─────────┐
    //               │                             │
    //               │                             │
    //           Fill Queue                    Read Queue
    //               │                             │
    //               │                             │
    //               └─────>  Background Thr >─────┘
    //
    // The next() function pops the examples available in the read queue and
    // swaps the read and fill queues once the read queue is empty. In parallel,
    // the background thread continuously pushes examples read from inner_ to
    // the fill queue.

    if (next_queue_.empty()) {
        ensure_prefetch_running();

        {
            std::unique_lock<std::mutex> queue_lock{queue_mutex_};

            read_queue_cond_.wait(queue_lock, [this] {
                return state_ != prefetch_state::running || !fill_queue_.empty();
            });

            if (state_ == prefetch_state::faulted)
                std::rethrow_exception(exception_ptr_);

            std::swap(next_queue_, fill_queue_);
        }

        fill_queue_cond_.notify_one();
    }

    if (next_queue_.empty())
        return std::nullopt;

    data d = std::move(next_queue_.front());

    next_queue_.pop_front();

    return d;
}

std::size_t
prefetched_data_source::skip(std::size_t num_examples)
{
    // TODO
    return inner_->skip(num_examples);
}

void
prefetched_data_source::reset()
{
    stop_prefetch();

    state_ = prefetch_state::not_running;

    fill_queue_.clear();
    next_queue_.clear();

    exception_ptr_ = {};

    inner_->reset();
}

void
prefetched_data_source::record_position(tape &t) const
{
    stop_prefetch();

    // TODO: Save read and fill queues.

    inner_->record_position(t);
}

void
prefetched_data_source::reload_position(tape &t)
{
    stop_prefetch();

    // TODO: Load read and fill queues.

    inner_->reload_position(t);
}

void
prefetched_data_source::ensure_prefetch_running()
{
    if (state_ != prefetch_state::not_running)
        return;

    state_ = prefetch_state::running;

    // TODO: signals
    prefetch_thread_ = std::thread{&prefetched_data_source::prefetch, this};
}

void
prefetched_data_source::prefetch()
{
    while (state_ == prefetch_state::running) {
        std::optional<data> d;
        try {
            d = inner_->next();
        } catch (const std::exception &) {
            exception_ptr_ = std::current_exception();
        }

        {
            std::unique_lock<std::mutex> queue_lock{queue_mutex_};

            fill_queue_cond_.wait(queue_lock, [this] {
                return should_stop_prefetch_ || fill_queue_.size() < num_examples_;
            });

            if (exception_ptr_) {
                state_ = prefetch_state::faulted;
            } else if (!d) {
                state_ = prefetch_state::eod;
            } else {
                fill_queue_.push_back(*std::move(d));

                if (should_stop_prefetch_)
                    state_ = prefetch_state::not_running;
            }
        }

        read_queue_cond_.notify_one();
    }
}

void
prefetched_data_source::stop_prefetch() const noexcept
{
    if (state_ == prefetch_state::not_running)
        return;

    {
        std::unique_lock<std::mutex> queue_lock{queue_mutex_};

        should_stop_prefetch_ = true;
    }

    fill_queue_cond_.notify_one();

    prefetch_thread_.join();

    should_stop_prefetch_ = false;
}

}  // namespace fairseq2::detail
