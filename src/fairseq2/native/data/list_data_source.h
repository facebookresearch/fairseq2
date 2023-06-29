// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <utility>
#include <vector>

#include "fairseq2/native/data/data_source.h"

namespace fairseq2::detail {

class list_data_source final : public data_source {
public:
    explicit
    list_data_source(std::vector<data> &&lst) noexcept
      : list_(std::move(lst)), iter_{list_.begin()}
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
    std::vector<data> list_;
    std::vector<data>::iterator iter_;
};

}  // namespace fairseq2::detail
