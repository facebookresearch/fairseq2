// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <utility>

#include "fairseq2n/data/data.h"
#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class constant_data_source final : public data_source {
public:
    explicit
    constant_data_source(data &&example, std::optional<std::string> key) noexcept
      : example_{std::move(example)}, key_{std::move(key)}
    {}

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
    data example_;
    std::optional<std::string> key_;
};

}  // namespace fairseq2n::detail
