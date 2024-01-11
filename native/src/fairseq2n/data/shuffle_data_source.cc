// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/shuffle_data_source.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>

#include "fairseq2n/utils/cast.h"

namespace fairseq2n::detail {

shuffle_data_source::shuffle_data_source(
    std::unique_ptr<data_source> &&inner, std::size_t shuffle_window, bool strict) noexcept
  : inner_{std::move(inner)}, strict_{strict}
{
    if (shuffle_window == 0)
        shuffle_window_ = std::numeric_limits<std::size_t>::max();
    else
        shuffle_window_ = shuffle_window;

    generator_ = at::globalContext().defaultGenerator(at::kCPU);
}

std::optional<data>
shuffle_data_source::next()
{
    if (shuffle_window_ == 1)
        return inner_->next();

    if (fill_buffer_) {
        // We have an upper limit on the reserved buffer size to avoid OOM
        // errors.
        buffer_.reserve(std::min(shuffle_window_, max_pre_alloc_size_));

        for (std::size_t i = 0; i < shuffle_window_; i++) {
            std::optional<data> maybe_example = inner_->next();
            if (!maybe_example)
                break;

            buffer_.push_back(*std::move(maybe_example));
        }

        buffer_pos_ = buffer_.begin();
        buffer_end_ = buffer_.end();

        fill_buffer_ = false;
    }

    if (buffer_pos_ == buffer_end_)
        return std::nullopt;

    // Instead of sampling per call, we shuffle per buffer which gives much
    // better hardware cache utilization and can have significant impact on
    // the runtime performance.
    if (buffer_pos_ == buffer_.begin())
        shuffle();

    data &buffered_example = *buffer_pos_;

    data output = std::move(buffered_example);

    // If we have not reached the end of `inner_`, fill the position of the
    // moved example with a new example.
    if (buffer_end_ == buffer_.end()) {
        std::optional<data> maybe_example = inner_->next();
        if (maybe_example) {
            buffered_example = *std::move(maybe_example);
        } else {
            // Mark this position, so that once we cycle back to it, we can
            // stop.
            buffer_end_ = buffer_pos_;
        }
    }

    if (++buffer_pos_ == buffer_.end())
        buffer_pos_ = buffer_.begin();

    return output;
}

void
shuffle_data_source::reset()
{
    buffer_.clear();

    buffer_pos_ = buffer_.begin();
    buffer_end_ = buffer_.end();

    fill_buffer_ = true;

    inner_->reset();
}

void
shuffle_data_source::record_position(tape &t) const
{
    if (strict_) {
        t.record(buffer_);

        t.record(buffer_pos_ - buffer_.begin());
        t.record(buffer_end_ - buffer_.begin());

        t.record(fill_buffer_);
    }

    inner_->record_position(t);
}

void
shuffle_data_source::reload_position(tape &t)
{
    if (strict_) {
        buffer_ = t.read<data_list>();

        buffer_pos_ = buffer_.begin() + t.read<std::ptrdiff_t>();
        buffer_end_ = buffer_.begin() + t.read<std::ptrdiff_t>();

        fill_buffer_ = t.read<bool>();
    } else {
        buffer_.clear();

        buffer_pos_ = buffer_.begin();
        buffer_end_ = buffer_.end();

        fill_buffer_ = true;
    }

    inner_->reload_position(t);
}

bool
shuffle_data_source::is_infinite() const noexcept
{
    return inner_->is_infinite();
}

void
shuffle_data_source::shuffle()
{
    using std::swap;

    std::lock_guard<std::mutex> guard{generator_.mutex()};

    auto *gen = at::check_generator<at::CPUGeneratorImpl>(generator_);

    std::size_t s = static_cast<std::size_t>(buffer_end_ - buffer_.begin());

    // Vanilla Fisher and Yates'.
    for (; s > 1; s--) {
        std::uint64_t r = gen->random64();

        std::size_t idx = conditional_cast<std::size_t>(r) % s;
        if (idx != s - 1)
            swap(buffer_[s - 1], buffer_[idx]);
    }
}

}  // namespace fairseq2n::detail
