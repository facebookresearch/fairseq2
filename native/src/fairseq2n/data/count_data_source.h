// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>

#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class count_data_source final : public data_source {
public:
    explicit
    count_data_source(
        std::int64_t start, std::int64_t step, std::optional<std::string> &&maybe_key) noexcept
      : start_{start}, step_{step}, counter_{start}, maybe_key_{std::move(maybe_key)}
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
    std::int64_t start_;
    std::int64_t step_;
    std::int64_t counter_;
    std::optional<std::string> maybe_key_;
};

}  // namespace fairseq2n::detail
