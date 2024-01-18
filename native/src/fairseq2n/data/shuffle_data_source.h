// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include <ATen/Generator.h>

#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class shuffle_data_source final : public data_source {
    static constexpr std::size_t max_pre_alloc_size_ = 100'000;

public:
    explicit
    shuffle_data_source(
        std::unique_ptr<data_source> &&inner, std::size_t shuffle_window, bool strict) noexcept;

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
    void
    shuffle();

private:
    std::unique_ptr<data_source> inner_;
    data_list buffer_{};
    data_list::iterator buffer_pos_ = buffer_.begin();
    data_list::iterator buffer_end_ = buffer_.end();
    std::size_t shuffle_window_;
    at::Generator generator_;
    bool strict_;
    bool fill_buffer_ = true;
};

}  // namespace fairseq2n::detail
