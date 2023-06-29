// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "fairseq2/native/data/data_source.h"

namespace fairseq2::detail {

class collated_data_source final : public data_source {
public:
    explicit
    collated_data_source(
        std::unique_ptr<data_source> &&inner, std::optional<std::int64_t> pad_idx) noexcept
      : inner_{std::move(inner)}, pad_idx_{pad_idx}
    {}

    std::optional<data>
    next() override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

private:
    data
    collate(const std::vector<data> &batch);

private:
    std::unique_ptr<data_source> inner_;
    std::optional<std::int32_t> pad_idx_;
};

}  // namespace fairseq2::detail
