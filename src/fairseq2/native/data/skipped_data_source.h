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

class skipped_data_source final : public data_source {
public:
    explicit
    skipped_data_source(std::unique_ptr<data_source> &&inner, std::size_t num_examples) noexcept
        : inner_{std::move(inner)}, num_examples_{num_examples}
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
    std::size_t num_examples_;
    bool skipped_ = false;
};

}  // namespace fairseq2::detail
