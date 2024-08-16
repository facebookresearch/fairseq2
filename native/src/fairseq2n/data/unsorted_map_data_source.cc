// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/unsorted_map_data_source.h"

#include <exception>

#include "fairseq2n/data/detail/thread.h"

namespace fairseq2n::detail {

unsorted_map_data_source::~unsorted_map_data_source()
{
    stop_thread_pool();
}

std::optional<data>
unsorted_map_data_source::next()
{
    if (buffer_size_ == 0) {
        std::optional<data> maybe_example{inner_->next()};
        if (maybe_example) {
            return map_fns_[0](std::move(*maybe_example));
        }
        return std::nullopt;
    }

    // We pop and return examples from the read queue until we drain it and
    // then swap it with the fill queue. In parallel, the background threads
    // continuously push examples read from inner data source to the fill
    // queue.
    if (next_queue_.empty()) {
        ensure_thread_pool_running();

        {
            std::unique_lock<std::mutex> queue_lock{queue_mutex_};

            read_queue_condition_.wait(queue_lock, [this]
            {
                return state_ != unsorted_map_state::running || 
                       !fill_queue_.empty();
            });

            if (state_ == unsorted_map_state::eod || 
                state_ == unsorted_map_state::faulted) {

                queue_lock.unlock();

                join_all_threads();

                if (state_ == unsorted_map_state::faulted) {
                    std::rethrow_exception(exception_ptr_);
                }
            }

            std::swap(next_queue_, fill_queue_);
        }

        fill_queue_condition_.notify_all();
    }

    if (next_queue_.empty())
        return std::nullopt;

    data example = std::move(next_queue_.front());

    next_queue_.pop_front();

    return example;
}

void
unsorted_map_data_source::reset(bool reset_rng)
{
    stop_thread_pool();

    if (state_ == unsorted_map_state::faulted)
        std::rethrow_exception(exception_ptr_);

    state_ = unsorted_map_state::not_running;

    fill_queue_.clear();
    next_queue_.clear();

    inner_->reset(reset_rng);
}

void
unsorted_map_data_source::record_position(tape &t, bool strict) const
{
    stop_thread_pool();

    if (state_ == unsorted_map_state::faulted)
        std::rethrow_exception(exception_ptr_);

    if (strict) {
        data_list fill_buffer{fill_queue_.begin(), fill_queue_.end()};
        data_list next_buffer{next_queue_.begin(), next_queue_.end()};

        t.record(fill_buffer);
        t.record(next_buffer);
    }

    inner_->record_position(t, strict);
}

void
unsorted_map_data_source::reload_position(tape &t, bool strict)
{
    stop_thread_pool();

    if (state_ == unsorted_map_state::faulted)
        std::rethrow_exception(exception_ptr_);

    state_ = unsorted_map_state::not_running;

    if (strict) {
        auto fill_buffer = t.read<data_list>();
        auto next_buffer = t.read<data_list>();

        fill_queue_.assign(fill_buffer.begin(), fill_buffer.end());
        next_queue_.assign(next_buffer.begin(), next_buffer.end());
    } else {
        fill_queue_.clear();
        next_queue_.clear();
    }

    inner_->reload_position(t, strict);
}

data_source_finitude_type
unsorted_map_data_source::finitude_type() const noexcept
{
    return inner_->finitude_type();
}

void
unsorted_map_data_source::ensure_thread_pool_running()
{
    {
        std::unique_lock<std::mutex> queue_lock{queue_mutex_};

        if (state_ == unsorted_map_state::eod || 
            state_ == unsorted_map_state::faulted)
            return;

        state_ = unsorted_map_state::running;

        for (std::size_t i = 0; i < thread_pool_.size(); ++i) {
            std::thread& thread = thread_pool_[i];
            if (!thread.joinable())
                thread = start_thread(&unsorted_map_data_source::fetch_and_map, this, i);
        }
    }
}

void
unsorted_map_data_source::fetch_and_map(std::size_t thread_idx)
{
    bool running = true;
    while (running) {
        std::optional<data> maybe_example{};
        std::exception_ptr exception_ptr;
        try {

            {
                std::unique_lock<std::mutex> pipeline_lock{pipeline_mutex_};
                maybe_example = inner_->next();
            }
            if (maybe_example) {
                maybe_example = map_fns_[thread_idx](std::move(*maybe_example));
            }

        } catch (const std::exception &) {
            exception_ptr = std::current_exception();
        }

        {
            std::unique_lock<std::mutex> queue_lock{queue_mutex_};

            fill_queue_condition_.wait(queue_lock, [this]
            {
                return state_ != unsorted_map_state::running || 
                       should_stop_pool_ || 
                       fill_queue_.size() < buffer_size_;
            });

            if (state_ != unsorted_map_state::running) {
                //Different thread has stopped thread pool

                if (exception_ptr) {
                    state_ = unsorted_map_state::faulted;
                    exception_ptr_ = exception_ptr;
                }
                else if (maybe_example) {
                    fill_queue_.push_back(*std::move(maybe_example));
                } 

                break;
            } else if (exception_ptr) {
                state_ = unsorted_map_state::faulted;
                exception_ptr_ = exception_ptr;
                running = false;
            } else if (!maybe_example) {
                state_ = unsorted_map_state::eod;
                running = false;
            } else {
                fill_queue_.push_back(*std::move(maybe_example));

                if (should_stop_pool_) {
                    state_ = unsorted_map_state::not_running;
                    running = false;
                }
            }
        }

        read_queue_condition_.notify_one();
    }
}

void
unsorted_map_data_source::stop_thread_pool() const noexcept
{
    {
        std::unique_lock<std::mutex> queue_lock{queue_mutex_};

        should_stop_pool_ = true;
    }

    fill_queue_condition_.notify_all();

    join_all_threads();

    should_stop_pool_ = false;
}

void
unsorted_map_data_source::join_all_threads() const noexcept
{
    for (std::thread& thread : thread_pool_) {
        if (thread.joinable())
            thread.join();
    }
}

}  // namespace fairseq2n::detail
