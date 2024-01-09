// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/prefetch_data_source.h"

#include <exception>

#include "fairseq2n/data/detail/thread.h"

namespace fairseq2n::detail {

prefetch_data_source::~prefetch_data_source()
{
    stop_prefetch_thread();
}

std::optional<data>
prefetch_data_source::next()
{
    // We pop and return examples from the read queue until we drain it and
    // then swap it with the fill queue. In parallel, the background thread
    // continuously pushes examples read from inner data source to the fill
    // queue.
    if (next_queue_.empty()) {
        ensure_prefetch_thread_running();

        {
            std::unique_lock<std::mutex> queue_lock{queue_mutex_};

            read_queue_condition_.wait(queue_lock, [this]
            {
                return state_ != prefetch_state::running || !fill_queue_.empty();
            });

            if (state_ == prefetch_state::faulted)
                std::rethrow_exception(exception_ptr_);

            std::swap(next_queue_, fill_queue_);
        }

        fill_queue_condition_.notify_one();
    }

    if (next_queue_.empty())
        return std::nullopt;

    data example = std::move(next_queue_.front());

    next_queue_.pop_front();

    return example;
}

void
prefetch_data_source::reset()
{
    stop_prefetch_thread();

    if (state_ == prefetch_state::faulted)
        std::rethrow_exception(exception_ptr_);

    state_ = prefetch_state::not_running;

    fill_queue_.clear();
    next_queue_.clear();

    inner_->reset();
}

void
prefetch_data_source::record_position(tape &t) const
{
    stop_prefetch_thread();

    if (state_ == prefetch_state::faulted)
        std::rethrow_exception(exception_ptr_);

    data_list fill_buffer{fill_queue_.begin(), fill_queue_.end()};
    data_list next_buffer{next_queue_.begin(), next_queue_.end()};

    t.record(fill_buffer);
    t.record(next_buffer);

    inner_->record_position(t);
}

void
prefetch_data_source::reload_position(tape &t)
{
    stop_prefetch_thread();

    if (state_ == prefetch_state::faulted)
        std::rethrow_exception(exception_ptr_);

    state_ = prefetch_state::not_running;

    auto fill_buffer = t.read<data_list>();
    auto next_buffer = t.read<data_list>();

    fill_queue_.assign(fill_buffer.begin(), fill_buffer.end());
    next_queue_.assign(next_buffer.begin(), next_buffer.end());

    inner_->reload_position(t);
}

bool
prefetch_data_source::is_infinite() const noexcept
{
    return inner_->is_infinite();
}

void
prefetch_data_source::ensure_prefetch_thread_running()
{
    if (state_ == prefetch_state::eod || state_ == prefetch_state::faulted)
        return;

    if (prefetch_thread_.joinable())
        return;

    state_ = prefetch_state::running;

    prefetch_thread_ = start_thread(&prefetch_data_source::prefetch, this);
}

void
prefetch_data_source::prefetch()
{
    while (state_ == prefetch_state::running) {
        std::optional<data> maybe_example{};
        try {
            maybe_example = inner_->next();
        } catch (const std::exception &) {
            exception_ptr_ = std::current_exception();
        }

        {
            std::unique_lock<std::mutex> queue_lock{queue_mutex_};

            fill_queue_condition_.wait(queue_lock, [this]
            {
                return should_stop_prefetch_ || fill_queue_.size() < num_examples_;
            });

            if (exception_ptr_) {
                state_ = prefetch_state::faulted;
            } else if (!maybe_example) {
                state_ = prefetch_state::eod;
            } else {
                fill_queue_.push_back(*std::move(maybe_example));

                if (should_stop_prefetch_)
                    state_ = prefetch_state::not_running;
            }
        }

        read_queue_condition_.notify_one();
    }
}

void
prefetch_data_source::stop_prefetch_thread() const noexcept
{
    if (!prefetch_thread_.joinable())
        return;

    {
        std::unique_lock<std::mutex> queue_lock{queue_mutex_};

        should_stop_prefetch_ = true;
    }

    fill_queue_condition_.notify_one();

    prefetch_thread_.join();

    should_stop_prefetch_ = false;
}

}  // namespace fairseq2n::detail
