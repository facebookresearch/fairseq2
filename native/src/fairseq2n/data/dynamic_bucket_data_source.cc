// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/dynamic_bucket_data_source.h"

#include <vector>

namespace fairseq2n::detail {

dynamic_bucket_data_source::dynamic_bucket_data_source(
    std::unique_ptr<data_source> &&inner, 
    float64 threshold, 
    cost_fn &&fn, 
    std::optional<bucket_creation_fn> &&maybe_bucket_fn,
    std::optional<std::size_t> maybe_min_num_examples, 
    std::optional<std::size_t> maybe_max_num_examples, 
    bool drop_remainder) noexcept
  : inner_{std::move(inner)}, 
    threshold_{threshold}, 
    cost_fn_{std::move(fn)}, 
    maybe_bucket_creation_fn_{std::move(maybe_bucket_fn)},
    maybe_min_num_examples_{maybe_min_num_examples}, 
    maybe_max_num_examples_{maybe_max_num_examples}, 
    drop_remainder_{drop_remainder}
{}

std::optional<data>
dynamic_bucket_data_source::next()
{
    if (!return_buffer_.empty()) {
        data output{return_buffer_.front()};
        return_buffer_.pop_front();
        return output;
    }

    if (maybe_min_num_examples_)
        buffer_.reserve(*maybe_min_num_examples_);

    float64 cost = 0;

    auto bucket_ready = [&]() {
        bool cost_threshold_met = cost >= threshold_;

        bool minimum_size_met = true;
        if (maybe_min_num_examples_)
            minimum_size_met = buffer_.size() >= *maybe_min_num_examples_;

        if (cost_threshold_met && minimum_size_met) return true;

        bool maximum_size_met = false;
        if (maybe_max_num_examples_)
            maximum_size_met = buffer_.size() >= *maybe_max_num_examples_;

        return maximum_size_met;
    };

    while (!bucket_ready()) {
        std::optional<data> maybe_example = inner_->next();
        if (!maybe_example)
            break;
        cost += cost_fn_(*maybe_example);
        buffer_.push_back(*std::move(maybe_example));
    }

    if (buffer_.empty())
        return std::nullopt;

    if (bucket_ready()) {
        if (maybe_bucket_creation_fn_) {
            const bucket_creation_fn& fn = *maybe_bucket_creation_fn_;
            auto&& [return_buffer, new_buffer] = fn(std::move(buffer_));

            buffer_ = std::move(new_buffer);

            data output{return_buffer.front()};
            return_buffer.pop_front();

            return_buffer_ = std::move(return_buffer);

            return output;
        } 
    } else if (drop_remainder_) {
        buffer_.clear();
        return std::nullopt;
    }

    data_list output = std::move(buffer_);
    buffer_.clear();
    return output;
}

void
dynamic_bucket_data_source::reset(bool reset_rng)
{
    buffer_.clear();
    inner_->reset(reset_rng);
}

void
dynamic_bucket_data_source::record_position(tape &t, bool strict) const
{
    if (maybe_bucket_creation_fn_) {
        t.record(buffer_);
    }
    inner_->record_position(t, strict);
}

void
dynamic_bucket_data_source::reload_position(tape &t, bool strict)
{
    if (maybe_bucket_creation_fn_) {
        buffer_ = t.read<data_list>();
    }
    inner_->reload_position(t, strict);
}

data_source_finitude_type
dynamic_bucket_data_source::finitude_type() const noexcept
{
    return inner_->finitude_type();
}

}  // namespace fairseq2n::detail
