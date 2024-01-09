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

class shard_data_source final : public data_source {
public:
    explicit
    shard_data_source(
        std::unique_ptr<data_source> &&inner,
        std::size_t shard_idx,
        std::size_t num_shards) noexcept
      : inner_{std::move(inner)}, shard_idx_{shard_idx}, num_shards_{num_shards}
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
    std::unique_ptr<data_source> inner_;
    std::size_t shard_idx_;
    std::size_t num_shards_;
};

}  // namespace fairseq2n::detail
