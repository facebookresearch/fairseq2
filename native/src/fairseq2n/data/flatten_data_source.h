// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>

#include "fairseq2n/data/data_source.h"
#include "fairseq2n/data/data.h"

namespace fairseq2n::detail {

class flatten_data_source final : public data_source {
public:
    explicit
    flatten_data_source(
        std::unique_ptr<data_source> &&inner,
        std::optional<std::string> selector) noexcept
      : inner_{std::move(inner)}, selector_{std::move(selector)}
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
    // Extracts elements from a data object based on the selector.
    // Returns a queue of elements to be returned by next().
    std::queue<data>
    extract_elements(const data &example);

    std::unique_ptr<data_source> inner_;
    std::optional<std::string> selector_;
    std::queue<data> elements_queue_;
    //std::exception_ptr exception_ptr_{};
};

}  // namespace fairseq2n::detail
