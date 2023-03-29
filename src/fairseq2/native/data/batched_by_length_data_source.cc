// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/batched_by_length_data_source.h"

#include <vector>


#include <iostream>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include "fairseq2/native/exception.h"
#include "fairseq2/native/data/data_pipeline.h"

namespace fairseq2::detail {

std::optional<data>
batched_by_length_data_source::next()
{
    auto n_buckets = bucket_sizes_.size();
    while (auto d = inner_->next()) {
        if (!d->is_tensor())
            throw data_pipeline_error{"The input to batch_by_length operation is expected to be a tensor."};

        auto example = d->as_tensor();
        auto seq_len = static_cast<std::size_t>(example.size(0));

        for (std::size_t i = 0; i < n_buckets; i++) {
            auto [bucket_batch_size, bucket_seq_len] = bucket_sizes_[i];
            if (seq_len <= bucket_seq_len) {
                buffers_[i].emplace_back(std::move(example));
                if (buffers_[i].size() >= bucket_batch_size) {
                    return make_batch(buffers_[i]);
                }
                break;
            }
        }
    }

    for (std::size_t i = 0; i < n_buckets; i++) {
        if (buffers_[i].empty())
            continue;

        return make_batch(buffers_[i]);
    }

    return std::nullopt;
}

at::Tensor
batched_by_length_data_source::make_batch(std::vector<at::Tensor>& batch) const
{
    auto tensor = at::pad_sequence(batch, /*batch_first=*/true, pad_idx_);
    batch.clear();
    return tensor;
}

std::size_t
batched_by_length_data_source::skip([[maybe_unused]] std::size_t num_examples)
{
    throw not_supported_error{"batch_by_length can't be efficiently sharded, you need to shard before calling 'batched_by_length'."};
}

void
batched_by_length_data_source::reset()
{
    inner_->reset();
}

void
batched_by_length_data_source::record_position(tape &t) const
{
    // TODO: we should also persist this.buffers_
    // https://github.com/fairinternal/fairseq2/issues/267
    inner_->record_position(t);
}

void
batched_by_length_data_source::reload_position(tape &t)
{
    // TODO: we should also reload this.buffers_
    // https://github.com/fairinternal/fairseq2/issues/267
    inner_->reload_position(t);
}

}  // namespace fairseq2::detail
