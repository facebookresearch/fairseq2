// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/batched_data_source.h"

#include <vector>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include "fairseq2/native/exception.h"

namespace fairseq2::detail {

std::optional<data>
batched_data_source::next()
{
    std::vector<data> batch{};

    batch.reserve(batch_size_);

    for (std::size_t i = 0; i < batch_size_; i++) {
        std::optional<data> d = inner_->next();
        if (!d)
            break;

        batch.emplace_back(*std::move(d));
    }

    if (batch.empty())
        return std::nullopt;

    if (drop_remainder_ && batch.size() < batch_size_)
        return std::nullopt;

    if (batch.front().is_py() || batch.front().is_string())
        return batch;

    if (batch.front().is_tensor()) {
        std::vector<at::Tensor> s{};

        s.reserve(batch.size());

        for (auto &v : batch)
            s.emplace_back(v.as_tensor());

        return at::stack(s);
    }

    throw not_supported_error{"The batch implementation is in-progress"};
}

std::size_t
batched_data_source::skip(std::size_t num_examples)
{
    return inner_->skip(num_examples * batch_size_) / batch_size_;
}

void
batched_data_source::reset()
{
    inner_->reset();
}

void
batched_data_source::record_position(tape &t) const
{
    inner_->record_position(t);
}

void
batched_data_source::reload_position(tape &t)
{
    inner_->reload_position(t);
}

}  // namespace fairseq2::detail
