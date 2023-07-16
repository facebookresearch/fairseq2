// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <utility>

#include "fairseq2/native/data/data.h"
#include "fairseq2/native/data/data_source.h"

namespace fairseq2::detail {

class constant_data_source final : public data_source {
public:
    explicit
    constant_data_source(data &&example) noexcept
      : example_{std::move(example)}
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
    data example_;
};

}  // namespace fairseq2::detail
