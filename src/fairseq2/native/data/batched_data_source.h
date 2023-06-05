// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "fairseq2/native/data/data_source.h"

namespace fairseq2::detail {

class batched_data_source final : public data_source {
public:
    explicit
    batched_data_source(std::unique_ptr<data_source> &&inner, std::size_t batch_size, bool drop_remainder, std::vector<std::int32_t> pad_idx) noexcept
        : inner_{std::move(inner)}, batch_size_{batch_size}, drop_remainder_{drop_remainder}, pad_idx_{std::move(pad_idx)}
    {
        if (pad_idx_.size() == 1)
            current_pad_idx_ = pad_idx_[0];
    }

    std::optional<data>
    next() override;

    std::size_t
    skip(std::size_t num_examples) override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

private:
    data
    make_batch(std::vector<data> batch);

    std::unique_ptr<data_source> inner_;
    std::size_t batch_size_;
    bool drop_remainder_;
    std::vector<std::int32_t> pad_idx_;
    std::optional<std::int32_t> current_pad_idx_;
};

}  // namespace fairseq2::detail
