// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/shuffled_data_source.h"

#include <algorithm>
#include <cstdint>
#include <mutex>

#include <ATen/Context.h>
#include <ATen/Generator.h>

#include "fairseq2/native/utils/cast.h"

namespace fairseq2::detail {

std::optional<data>
shuffled_data_source::next()
{
    if (shuffle_window_ == 1)
        return inner_->next();

    if (fill_buffer_) {
        // We have an upper limit on the reserved buffer size to avoid OOM
        // errors.
        buffer_.reserve(std::min(shuffle_window_, max_pre_alloc_size_));

        for (std::size_t i = 0; i < shuffle_window_; i++) {
            std::optional<data> d = inner_->next();
            if (!d)
                break;

            buffer_.push_back(*std::move(d));
        }

        fill_buffer_ = false;
    }

    if (buffer_.empty())
        return std::nullopt;

    // Pick an index from a uniform distribution.
    std::size_t idx = random_index();

    data &picked_element = buffer_[idx];

    data output = std::move(picked_element);

    // Fill the position of the moved element with a new example.
    std::optional<data> d = inner_->next();
    if (d) {
        picked_element = *std::move(d);
    } else {
        // If we can't fill the position with a new example, it means we reached
        // the end of data; start shrinking the size of the buffer.
        picked_element = std::move(buffer_.back());

        buffer_.pop_back();
    }

    return output;
}

void
shuffled_data_source::reset()
{
    buffer_.clear();

    fill_buffer_ = true;

    inner_->reset();
}

void
shuffled_data_source::record_position(tape &t) const
{
    if (strict_) {
        t.record(buffer_);

        t.record(fill_buffer_);
    }

    inner_->record_position(t);
}

void
shuffled_data_source::reload_position(tape &t)
{
    if (strict_) {
        buffer_ = t.read<std::vector<data>>();

        fill_buffer_ = t.read<bool>();
    } else {
        buffer_.clear();

        fill_buffer_ = true;
    }

    inner_->reload_position(t);
}

std::size_t
shuffled_data_source::random_index() const
{
    at::Generator g = at::globalContext().defaultGenerator(at::kCPU);

    std::lock_guard<std::mutex> g_lock{g.mutex()};

    auto *cpu_generator = at::check_generator<at::CPUGeneratorImpl>(g);

    std::uint64_t nr = cpu_generator->random64();

    return conditional_cast<std::size_t>(nr) % buffer_.size();
}

}  // namespace fairseq2::detail
