// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "fairseq2/native/exception.h"
#include "fairseq2/native/data/data_source.h"

namespace fairseq2::detail {



class batched_by_length_data_source final : public data_source {
public:
    explicit
    batched_by_length_data_source(
        std::unique_ptr<data_source> &&inner,
        std::vector<std::pair<std::size_t, std::size_t>> bucket_sizes,
        std::int32_t pad_idx
    ) : inner_(std::move(inner)),
        bucket_sizes_(std::move(bucket_sizes)),
        pad_idx_(pad_idx)
    {
        std::sort(bucket_sizes_.begin(), bucket_sizes_.end(), [](auto x, auto y){return x.second < y.second;});
        buffers_.reserve(bucket_sizes_.size());
        for (auto &size : bucket_sizes_) {
            buffers_.emplace_back().reserve(size.first);
        }
    }

    std::optional<data>
    next() override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

private:
    at::Tensor
    make_batch(std::vector<at::Tensor>& batch) const;

    std::unique_ptr<data_source> inner_;
    std::vector<std::pair<std::size_t, std::size_t>> bucket_sizes_;
    std::int32_t pad_idx_;
    std::vector<std::vector<at::Tensor>> buffers_{};
};

}  // namespace fairseq2::detail
