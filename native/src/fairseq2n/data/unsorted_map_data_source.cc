// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/unsorted_map_data_source.h"

#include <exception>

#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/detail/exception.h"
#include "fairseq2n/detail/parallel.h"

namespace fairseq2n::detail {

unsorted_map_data_source::unsorted_map_data_source(
    std::unique_ptr<data_source> &&inner, std::vector<map_fn> &&fns, std::size_t num_parallel_calls)
  : inner_{std::move(inner)},
    map_fns_{std::move(fns)},
    num_parallel_calls_{num_parallel_calls},
    thread_pool_{num_parallel_calls},
    buffer_{std::vector(num_parallel_calls)}
{}

std::optional<data>
unsorted_map_data_source::next()
{
    if (num_parallel_calls_ <= 1) {
        while (std::optional<data> maybe_example = inner_->next()) {
            maybe_example = invoke_function(*std::move(maybe_example), 0);
            if (maybe_example)
                return maybe_example;
        }

        return std::nullopt;
    }
    
    if (faulted_) {
        std::rethrow_exception(exception_ptr_);
    }

    ensure_thread_pool_running();

    return fill_buffer();
}

void
unsorted_map_data_source::reset(bool reset_rng)
{
    faulted_ = false;

    for (auto& element : buffer_) {
        element.store(std::nullopt);
    }

    inner_->reset(reset_rng);
}

void
unsorted_map_data_source::record_position(tape &t, bool strict) const
{
    stop_thread_pool();

    if (faulted_) {
        std::rethrow_exception(exception_ptr_);
    }

    if (strict) {
        data_list result_queue_buffer;
        result_queue_buffer.reserve(result_queue_.size());
        for (auto& result : result_queue_) {
            if (std::holds_alternative<std::exception_ptr>(result)) {
                faulted_ = true;
                exception_ptr_ = std::get<std::exception_ptr>(result);
                std::rethrow_exception(exception_ptr_);
            }
            result_queue_buffer.push_back(std::get<data>(result));
        }

        t.record(result_queue_buffer);

        std::vector<std::size_t> task_queue_buffer;
        task_queue_buffer.reserve(task_queue_.size());
        for (auto& it : task_queue_) {
            task_queue_buffer.push_back(it - task_queue_.begin());
        }

        t.record(task_queue_buffer);

        std::vector<std::optional<data>> buffer;
        buffer.reserve(buffer_.size());
        for (auto& element : buffer_) {
            buffer.push_back(element.load());
        }

        t.record(buffer);
    }

    inner_->record_position(t, strict);
}

void
unsorted_map_data_source::reload_position(tape &t, bool strict)
{
    stop_thread_pool();

    result_queue_.clear();
    task_queue_.clear();

    if (strict) {

        for (const data& result : t.read<data_list>()) {
            result_queue_.push(result);
        }

        for (const auto& index : t.read<std::vector<std::size_t>>()) {
            task_queue_.push(buffer_.begin() + index);
        }

        std::vector<std::optional<data>> buffer{
            t.read<std::vector<std::optional<data>>>()
        };
        for (std::size_t i = 0; i < num_parallel_calls_; ++i) {
            buffer_[i].store(buffer[i]);
        }
    } else {
        buffer_.clear();
    }

    inner_->reload_position(t, strict);
}

data_source_finitude_type
unsorted_map_data_source::finitude_type() const noexcept
{
    return inner_->finitude_type();
}

std::optional<data>
unsorted_map_data_source::fill_buffer()
{
    bool empty_buffer = true;
    for (auto it = buffer_.begin(); it != buffer_.end(); ++it) {
        if (*it) { 
            empty_buffer = false;
            continue;
        }

        std::optional<data> maybe_example;
        try {
            maybe_example = inner_->next();
        } catch (const std::exception &) {
            stop_thread_pool();
            faulted_ = true;
            exception_ptr_ = std::current_exception();
            std::rethrow_exception(std::current_exception());
        }
        if (!maybe_example)
            break;

        {
            std::unique_lock<std::mutex> queue_lock{task_queue_mutex_};
            task_queue_.push(it);
        }
        task_queue_condition_.notify_one();
    }

    if (empty_buffer)
        return std::nullopt;

    
    std::unique_lock<std::mutex> queue_lock{result_queue_mutex_};

    result_queue_condition_.wait(queue_lock, [this]
    {
        return !fill_queue_.empty();
    });

    auto result = result_queue_.front()->load();
    if (std::holds_alternative<std::exception_ptr>(result)) {
        stop_thread_pool();
        faulted_ = true;
        exception_ptr_ = std::get<std::exception_ptr>(result);
        std::rethrow_exception(exception_ptr_);
    }
    result_queue_.pop();
    return std::get<data>(result);
}

std::optional<data>
unsorted_map_data_source::invoke_function(data &&example, std::size_t fn_idx)
{
    return map_fns_[fn_idx](std::move(example));
}

void
unsorted_map_data_source::run_map()
{
    while (true) {
        std::optional<data> maybe_example;
        std::exception_ptr exception_ptr;
        std::vector<std::atomic<std::optional<data>>>::iterator buffer_iterator;
        {
            std::unique_lock<std::mutex> queue_lock{task_queue_mutex_};

            task_queue_condition_.wait(queue_lock, [this]
            {
                return should_stop_map_ || !task_queue_.empty();
            });

            if (should_stop_map_) {
                break;
            }
            
            buffer_iterator = task_queue_.front();
            maybe_example = buffer_iterator->load();
            task_queue_.pop();
        }

        if (!maybe_example) {
            //Should never happen
            break;
        }

        std::size_t buffer_pos = buffer_iterator - buffer_.begin();
        try {
            maybe_example = invoke_function(*std::move(maybe_example), buffer_pos);
        } catch (const std::exception&) {
            exception_ptr = std::current_exception();
        }

        {
            std::unique_lock<std::mutex> queue_lock{result_queue_mutex_};
            if (maybe_example) {
                result_queue_.push(*std::move(maybe_example));
            } else {
                result_queue_.push(exception_ptr);
            }
        }

        buffer_iterator->store(std::nullopt);

        result_queue_condition_.notify_one();
    }
}

void
unsorted_map_data_source::ensure_thread_pool_running()
{
    for (auto& thread : thread_pool_) {
        if (prefetch_thread_.joinable())
            continue;
        thread = start_thread(&unsorted_map_data_source::run_map, this);
    }
}

void
stop_thread_pool()
{
    {
        std::unique_lock<std::mutex> task_queue_lock{task_queue_mutex_};
        std::unique_lock<std::mutex> result_queue_lock{result_queue_mutex_};

        should_stop_map_ = true;
    }

    task_queue_condition_.notify_all();
    result_queue_condition_.notify_all();

    for (auto& thread : thread_pool_) {
        thread.join();
    }

    should_stop_map_ = false;
}

}  // namespace fairseq2n::detail
