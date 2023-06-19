// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/mapped_data_source.h"

#include <oneapi/tbb.h>

namespace fairseq2::detail {

std::optional<data>
mapped_data_source::next()
{
    if (num_parallel_calls_ <= 1) {
        std::optional<data> d = inner_->next();
        if (!d)
            return std::nullopt;

        return invoke_map_fn(*std::move(d));
    }

    // If we have exhausted all buffered examples, try to refill the buffer.
    if (buffer_iter_ == buffer_.end() && !fill_buffer())
        return std::nullopt;

    // Yield a buffered example.
    return std::move(*buffer_iter_++);
}

void
mapped_data_source::reset()
{
    buffer_.clear();

    buffer_iter_ = buffer_.begin();

    inner_->reset();
}

void
mapped_data_source::record_position(tape &t) const
{
    t.record(buffer_);

    t.record(buffer_iter_ - buffer_.begin());

    inner_->record_position(t);
}

void
mapped_data_source::reload_position(tape &t)
{
    buffer_ = t.read<std::vector<data>>();

    buffer_iter_ = buffer_.begin() + t.read<std::ptrdiff_t>();

    inner_->reload_position(t);
}

bool
mapped_data_source::fill_buffer()
{
    buffer_.clear();

    for (std::size_t i = 0; i < num_parallel_calls_; i++) {
        std::optional<data> d = inner_->next();
        if (!d)
            break;

        buffer_.push_back(*std::move(d));
    }

    if (buffer_.empty())
        return false;

    // Apply the map function to all buffered examples.
    auto apply_map_fn = [this](const tbb::blocked_range<std::size_t> &rng) {
        for (auto i = rng.begin(); i < rng.end(); ++i)
            buffer_[i] = invoke_map_fn(std::move(buffer_[i]));
    };

    tbb::blocked_range<std::size_t> full_rng{0, buffer_.size()};

    // Avoid threading overhead if we have just one example.
    if (buffer_.size() == 1)
        apply_map_fn(full_rng);
    else
        tbb::parallel_for(full_rng, apply_map_fn);

    buffer_iter_ = buffer_.begin();

    return true;
}

data
mapped_data_source::invoke_map_fn(data &&example) {
    try {
        return fn_(std::move(example));
    } catch (const data_pipeline_error &) {
        throw;
    } catch (...) {
        data_pipeline_error::throw_nested("The map function has failed.");
    }
}

}  // namespace fairseq2::detail
