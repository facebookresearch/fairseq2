// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class bucket_data_source final : public data_source {
public:
    explicit
    bucket_data_source(
        std::unique_ptr<data_source> &&inner,
        std::size_t bucket_size,
        bool drop_remainder) noexcept;

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
    std::size_t bucket_size_;
    bool drop_remainder_;
};

}  // namespace fairseq2n::detail
