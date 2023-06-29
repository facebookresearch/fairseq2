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

#include "fairseq2/native/data/data_source.h"
#include "fairseq2/native/data/element_selector.h"

namespace fairseq2::detail {

class batched_by_length_data_source final : public data_source {
public:
    explicit
    batched_by_length_data_source(
        std::unique_ptr<data_source> &&inner,
        std::vector<std::pair<std::size_t, std::size_t>> &&bucket_sizes,
        std::size_t max_seq_len,
        std::optional<element_selector> &&selector,
        bool drop_remainder,
        bool warn_only);

    std::optional<data>
    next() override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

private:
    std::optional<std::size_t>
    determine_seq_len(const data &d);

private:
    std::unique_ptr<data_source> inner_;
    std::vector<std::pair<std::size_t, std::size_t>> bucket_sizes_;
    std::size_t max_seq_len_;
    std::optional<element_selector> selector_{};
    bool drop_remainder_;
    bool warn_only_;
    std::vector<std::vector<data>> buckets_{};
};

}  // namespace fairseq2::detail
