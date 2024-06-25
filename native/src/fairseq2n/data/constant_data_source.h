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
    constant_data_source(data &&example, std::optional<std::string> &&maybe_key) noexcept
      : example_{std::move(example)}, maybe_key_{std::move(maybe_key)}
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
    data example_;
    std::optional<std::string> maybe_key_;
};

}  // namespace fairseq2n::detail
