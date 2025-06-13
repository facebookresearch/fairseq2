// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include <ATen/Tensor.h>

#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class packed_data_source final : public data_source {
public:
    explicit
    packed_data_source(
        std::unique_ptr<data_source> &&inner,
        std::int64_t num_elements,
        std::int64_t max_seq_len,
        std::int64_t pad_value,
        bool truncate,
        bool drop_remainder,
        bool pinned_memory
    ) noexcept
      : inner_{std::move(inner)},
        capacity_{num_elements},
        max_seq_len_{max_seq_len},
        pad_value_{pad_value},
        remainder_{},
        truncate_{truncate},
        drop_remainder_{drop_remainder},
        pinned_memory_{pinned_memory}
    {}

    std::optional<data>
    next() override;

    void
    reset(bool reset_rng) override;

    void
    record_position(tape &t, bool strict) const override;

    void
    reload_position(tape &t, bool strict) override;

    data_source_finitude_type
    finitude_type() const noexcept override;

private:
    std::unique_ptr<data_source> inner_;
    std::int64_t capacity_;
    std::int64_t max_seq_len_;
    std::int64_t pad_value_;
    at::Tensor remainder_;
    bool truncate_;
    bool drop_remainder_;
    bool pinned_memory_;
};

}  // namespace fairseq2n::detail
