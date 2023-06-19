// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "fairseq2/native/data/data_source.h"

namespace fairseq2::detail {

class ranged_data_source final : public data_source {
public:
    explicit
    ranged_data_source(std::unique_ptr<data_source> &&inner, std::size_t count) noexcept
        : inner_{std::move(inner)}, count_{count}
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
    std::unique_ptr<data_source> inner_;
    std::size_t count_;
    std::size_t num_examples_read_ = 0;
};

}  // namespace fairseq2::detail
