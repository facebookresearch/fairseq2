// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/map_data_source.h"

#include <exception>

#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/detail/exception.h"
#include "fairseq2n/detail/parallel.h"
#include "fairseq2n/utils/cast.h"

#ifdef FAIRSEQ2N_USE_TBB
using PoolArgType = int;
#include <oneapi/tbb.h>
#else
using PoolArgType = std::size_t;
#endif

namespace fairseq2n::detail {

map_data_source::map_data_source(
    std::unique_ptr<data_source> &&inner,
    std::vector<map_fn> &&fns,
    std::size_t num_parallel_calls,
    bool deterministic)
    : inner_{std::move(inner)},
      map_fns_{std::move(fns)},
      num_parallel_calls_{num_parallel_calls},
      deterministic_{deterministic || num_parallel_calls == 1},
      pool_{conditional_cast<PoolArgType>(deterministic ? 0U : num_parallel_calls)}
{
    buffer_.reserve(num_parallel_calls);

    buffer_pos_ = buffer_.begin();
}

std::optional<data>
map_data_source::next()
{
    if (num_parallel_calls_ <= 1) {
        while (std::optional<data> maybe_example = inner_->next()) {
            maybe_example = invoke_function(*std::move(maybe_example), 0);
            if (maybe_example)
                return maybe_example;
        }

        return std::nullopt;
    }

    if (deterministic_) {
        do {
            // Yield a buffered example.
            for (; buffer_pos_ < buffer_.end(); ++buffer_pos_) {
                if (buffer_pos_->has_value()) {
                    std::optional<data> output = std::exchange(*buffer_pos_, std::nullopt);

                    ++buffer_pos_;

                    return output;
                }
            }
            // If we have exhausted all buffered examples, try to refill the buffer.
        } while (fill_buffer());
    } else {
        // Check that we either have work or waiting outputs
        while (fill_buffer_async()) {
            // Wait until the next output is ready
            std::unique_lock<std::mutex> lock{async_output_mutex_};
            read_output_condition_.wait(lock, [this] {
                return !async_queue_.empty() || exception_ptr_;
            });

            if (exception_ptr_)
                std::rethrow_exception(exception_ptr_);

            auto example = std::move(async_queue_.front());
            async_queue_.pop_front();
            if (example)
                return example;
        }
    }

    return std::nullopt;
}

void
map_data_source::reset(bool reset_rng)
{
    buffer_.clear();

    buffer_pos_ = buffer_.begin();

    reset_async_state();

    async_queue_.clear();

    inner_->reset(reset_rng);
}

void
map_data_source::record_position(tape &t, bool strict) const
{
    if (strict) {
        if (deterministic_) {
            t.record(buffer_);

            t.record(buffer_pos_ - buffer_.begin());
        } else {
            // Wait until all current tasks have output to the queue
            wait_until_done();
            // Write the queue on the tape
            {
                std::unique_lock<std::mutex> lock{async_output_mutex_};
                t.record(async_queue_.size());

                for (const auto &element : async_queue_)
                    t.record(element);
            }
        }
    }

    inner_->record_position(t, strict);
}

void
map_data_source::reload_position(tape &t, bool strict)
{
    if (strict && deterministic_) {
        buffer_ = t.read<std::vector<std::optional<data>>>();

        buffer_pos_ = buffer_.begin() + t.read<std::ptrdiff_t>();
    } else if (strict && !deterministic_) {
        // Wait for all tasks to complete and reset state
        reset_async_state();

        async_queue_.clear();

        // Fill the queue again from the tape
        auto size = t.read<std::size_t>();
        for (std::size_t i = 0; i < size; ++i)
            async_queue_.push_back(t.read<std::optional<data>>());

        buffer_.clear();
        buffer_pos_ = buffer_.begin();
    } else {
        buffer_.clear();

        buffer_pos_ = buffer_.begin();

        reset_async_state();

        async_queue_.clear();
    }

    inner_->reload_position(t, strict);
}

data_source_finitude_type
map_data_source::finitude_type() const noexcept
{
    return inner_->finitude_type();
}

bool
map_data_source::fill_buffer()
{
    buffer_.clear();

    for (std::size_t i = 0; i < num_parallel_calls_; i++) {
        std::optional<data> maybe_example = inner_->next();
        if (!maybe_example)
            break;

        buffer_.push_back(std::move(maybe_example));
    }

    buffer_pos_ = buffer_.begin();

    if (buffer_.empty())
        return false;

    // Apply the processor to all buffered examples.
    auto apply_function = [this](std::size_t begin, std::size_t end) {
        for (auto i = begin; i < end; ++i)
            buffer_[i] = invoke_function(*std::move(buffer_[i]), i);
    };

    // Avoid threading overhead if we have just one example.
    if (buffer_.size() == 1)
        apply_function(0, buffer_.size());
    else
        parallel_for<std::size_t>(apply_function, buffer_.size());

    return true;
}

bool
map_data_source::has_async_output()
{
    std::unique_lock<std::mutex> lock(async_output_mutex_);
    return !async_queue_.empty();
}

void
map_data_source::reset_async_state()
{
    wait_until_done();

    finished_ = false;
}

void
map_data_source::wait_until_done() const
{
    std::unique_lock<std::mutex> lock{async_output_mutex_};
    read_output_condition_.wait(lock, [this] {
        return tasks_in_flight_ == 0 || exception_ptr_;
    });

    if (exception_ptr_)
        std::rethrow_exception(exception_ptr_);
}

void
map_data_source::AsyncMapTask::operator()() const
{
    try {
        // Move out of example, even though we're in a const operator
        data result = p_this->map_fns_[0](std::move(example));

        {
            std::unique_lock<std::mutex> lock(p_this->async_output_mutex_);
            p_this->async_queue_.emplace_back(std::move(result));
        }
    } catch (...) {
        std::unique_lock<std::mutex> lock(p_this->async_output_mutex_);
        p_this->exception_ptr_ = std::current_exception();
    }

    p_this->tasks_in_flight_--;
    p_this->read_output_condition_.notify_one();
}

bool
map_data_source::fill_buffer_async()
{
    for (std::size_t i = tasks_in_flight_; i < num_parallel_calls_; i++) {
        std::optional<data> maybe_example = inner_->next();
        if (!maybe_example) {
            finished_ = true;
            break;
        }

        tasks_in_flight_++;

        // Create task and send to thread pool
        data example = std::move(*maybe_example);

        pool_.enqueue(AsyncMapTask{this, std::move(example)});
    }

    return !finished_ || tasks_in_flight_ > 0 || has_async_output();
}

std::optional<data>
map_data_source::invoke_function(data &&example, std::size_t fn_idx)
{
    return map_fns_[fn_idx](std::move(example));
}

}  // namespace fairseq2n::detail
