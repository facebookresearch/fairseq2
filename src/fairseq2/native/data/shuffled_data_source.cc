// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/shuffled_data_source.h"

#include <cassert>

#include <ATen/Tensor.h>
#include <ATen/CPUGeneratorImpl.h>

#include "fairseq2/native/exception.h"


namespace fairseq2::detail {

std::optional<data>
shuffled_data_source::next()
{
    auto rng = rng_.get<at::CPUGeneratorImpl>();
    std::optional<data> d;
    while((d = inner_->next())) {
        std::size_t rand = rng->random64();
        // TODO: should we force user to have a power of 2 buffer_size ?
        std::size_t i = (rand >> 1) % buffer_.size();
        auto old = buffer_[i];
        if (!old) {
            // No item in bucket, just insert.
            buffer_[i] = std::move(d);
            continue;
        }
        // 50% chance of returning the old example already in bucket,
        // 50% chance of returning the new example.
        if ((bool)(rand & 1)) return d;
        data x = std::move(*old);
        buffer_[i] = std::move(d);
        return x;
    }

    while (remaining_off_ < buffer_.size()) {
        d = buffer_[remaining_off_++];
        if (!d) continue;
        return std::move(*d);
    }

    return {};
}

std::size_t
shuffled_data_source::skip(std::size_t num_examples)
{
    // For sharding we just want the different workers to have different examples.
    return inner_->skip(num_examples);
}

void
shuffled_data_source::reset()
{
    inner_->reset();
    remaining_off_ = 0;
    for (auto &buff: buffer_)
        buff = std::nullopt;
}

void
shuffled_data_source::record_position(tape &t) const
{
    inner_->record_position(t);
    t.record(deterministic_);
    if (!deterministic_) return;

    t.record(rng_.get_state());
    t.record(buffer_.size());
    t.record(remaining_off_);
    for (std::size_t i = remaining_off_; i < buffer_.size(); ++i)
        t.record(buffer_[i]);
}

void
shuffled_data_source::reload_position(tape &t)
{
    inner_->reload_position(t);
    deterministic_ = t.read<bool>();
    if (!deterministic_) return;

    rng_.set_state(t.read<at::Tensor>());
    auto buffer_size = t.read<std::size_t>();
    assert(buffer_size == buffer_.size());
    remaining_off_ = t.read<std::size_t>();
    for (std::size_t i = remaining_off_; i < buffer_size; ++i)
        buffer_[i] = t.read<std::optional<data>>();
}

}  // namespace fairseq2::detail
