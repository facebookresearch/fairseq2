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

namespace fairseq2n::detail {

map_data_source::map_data_source(
    std::unique_ptr<data_source> &&inner, std::vector<map_fn> &&fns, std::size_t num_parallel_calls)
  : inner_{std::move(inner)},
    map_fns_{std::move(fns)},
    num_parallel_calls_{num_parallel_calls}
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

    do {
        // Yield a buffered example.
        for (; buffer_pos_ < buffer_.end(); ++buffer_pos_) {
            if (*buffer_pos_)
                return std::move(*buffer_pos_++);
        }
    // If we have exhausted all buffered examples, try to refill the buffer.
    } while (fill_buffer());

    return std::nullopt;
}

void
map_data_source::reset(bool reset_rng)
{
    buffer_.clear();

    buffer_pos_ = buffer_.begin();

    inner_->reset(reset_rng);
}

void
map_data_source::record_position(tape &t, bool strict) const
{
    if (strict) {
        t.record(buffer_);

        t.record(buffer_pos_ - buffer_.begin());
    }

    inner_->record_position(t, strict);
}

void
map_data_source::reload_position(tape &t, bool strict)
{
    if (strict) {
        buffer_ = t.read<std::vector<std::optional<data>>>();

        buffer_pos_ = buffer_.begin() + t.read<std::ptrdiff_t>();
    } else {
        buffer_.clear();

        buffer_pos_ = buffer_.begin();
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

    if (buffer_.empty())
        return false;

    // Apply the processor to all buffered examples.
    auto apply_function = [this](std::size_t begin, std::size_t end)
    {
        for (auto i = begin; i < end; ++i)
            buffer_[i] = invoke_function(*std::move(buffer_[i]), i);
    };

    // Avoid threading overhead if we have just one example.
    if (buffer_.size() == 1)
        apply_function(0, buffer_.size());
    else
        parallel_for<std::size_t>(apply_function, buffer_.size());

    buffer_pos_ = buffer_.begin();

    return true;
}

std::optional<data>
map_data_source::invoke_function(data &&example, std::size_t fn_idx)
{
    return map_fns_[fn_idx](std::move(example));
}

}  // namespace fairseq2n::detail
