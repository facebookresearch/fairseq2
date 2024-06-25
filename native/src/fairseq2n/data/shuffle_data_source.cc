// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/shuffle_data_source.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>

#include "fairseq2n/data/detail/rng.h"
#include "fairseq2n/utils/cast.h"

namespace fairseq2n::detail {

shuffle_data_source::shuffle_data_source(
    std::unique_ptr<data_source> &&inner,
    std::size_t shuffle_window,
    std::optional<std::uint64_t> maybe_seed)
  : inner_{std::move(inner)}
{
    if (shuffle_window == 0)
        shuffle_window_ = std::numeric_limits<std::size_t>::max();
    else
        shuffle_window_ = shuffle_window;

    seed_ = maybe_seed ? *maybe_seed : pseudo_random();

    generator_ = at::make_generator<at::CPUGeneratorImpl>(seed_);
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
shuffle_data_source::reset(bool reset_rng)
{
    buffer_.clear();

    buffer_pos_ = buffer_.begin();
    buffer_end_ = buffer_.end();

    fill_buffer_ = true;

    if (reset_rng)
        generator_.set_current_seed(seed_);

    inner_->reset(reset_rng);
}

void
shuffle_data_source::record_position(tape &t, bool strict) const
{
    if (strict) {
        t.record(buffer_);

        t.record(buffer_pos_ - buffer_.begin());
        t.record(buffer_end_ - buffer_.begin());

        t.record(fill_buffer_);
    }

    t.record(seed_);

    t.record(generator_.get_state());

    inner_->record_position(t, strict);
}

void
shuffle_data_source::reload_position(tape &t, bool strict)
{
    if (strict) {
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

    seed_ = t.read<std::uint64_t>();

    generator_.set_state(t.read<at::Tensor>());

    inner_->reload_position(t, strict);
}

data_source_finitude_type
shuffle_data_source::finitude_type() const noexcept
{
    return inner_->finitude_type();
}

void
shuffle_data_source::shuffle()
{
    using std::swap;

    auto *gen = generator_.get<at::CPUGeneratorImpl>();

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
