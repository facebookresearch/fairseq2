// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <utility>
#include <vector>

#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class list_data_source final : public data_source {
public:
    explicit
    list_data_source(data_list &&list) noexcept
      : list_(std::move(list)), pos_{list_.begin()}
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
    data_list list_;
    data_list::iterator pos_;
};

}  // namespace fairseq2n::detail
