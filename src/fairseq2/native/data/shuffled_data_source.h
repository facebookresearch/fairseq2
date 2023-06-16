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
#include "ATen/core/Generator.h"
#include "ATen/CPUGeneratorImpl.h"

namespace fairseq2::detail {

class shuffled_data_source final : public data_source {
public:
    explicit
    shuffled_data_source(std::unique_ptr<data_source> &&inner, std::size_t buffer_size, std::size_t seed, bool deterministic)
        : inner_{std::move(inner)}, buffer_(buffer_size > 0 ? buffer_size : 1), deterministic_{deterministic}
    {
        rng_.set_current_seed(seed);
    }

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
    at::Generator rng_ = at::make_generator<at::CPUGeneratorImpl>();
    std::vector<std::optional<data>> buffer_;
    bool deterministic_;

    std::size_t remaining_off_ = 0;
};

}  // namespace fairseq2::detail
