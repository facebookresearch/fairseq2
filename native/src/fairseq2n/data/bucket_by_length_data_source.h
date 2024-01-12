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
        bool skip_long_examples,
        bool drop_remainder);

    std::optional<data>
    next() override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

    bool
    is_infinite() const noexcept override;

private:
    std::unique_ptr<data_source> inner_;
    std::vector<std::pair<std::size_t, std::size_t>> bucket_sizes_;
    std::size_t max_data_len_;
    data_length_fn data_length_fn_;
    bool skip_long_examples_;
    bool drop_remainder_;
    std::vector<data_list> buckets_{};
};

}  // namespace fairseq2n::detail
