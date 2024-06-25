// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <optional>
#include <memory>
#include <utility>
#include <vector>

#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class bucket_by_length_data_source final : public data_source {
public:
    explicit
    bucket_by_length_data_source(
        std::unique_ptr<data_source> &&inner,
        std::vector<std::pair<std::size_t, std::size_t>> &&bucket_sizes,
        data_length_fn &&fn,
        std::size_t min_data_len,
        bool skip_below_min_examples,
        bool skip_above_max_examples,
        bool drop_remainder);

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
    std::vector<std::pair<std::size_t, std::size_t>> bucket_sizes_;
    data_length_fn data_length_fn_;
    std::size_t min_data_len_;
    std::size_t max_data_len_;
    bool skip_below_min_examples_;
    bool skip_above_max_examples_;
    bool drop_remainder_;
    std::vector<data_list> buckets_{};
};

}  // namespace fairseq2n::detail
