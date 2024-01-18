// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <optional>

#include "fairseq2n/api.h"
#include "fairseq2n/data/data.h"
#include "fairseq2n/data/tape.h"

namespace fairseq2n {

class FAIRSEQ2_API data_source {
public:
    data_source() noexcept = default;

    data_source(const data_source &) = default;
    data_source &operator=(const data_source &) = default;

    data_source(data_source &&) = default;
    data_source &operator=(data_source &&) = default;

    virtual
   ~data_source();

    virtual std::optional<data>
    next() = 0;

    virtual void
    reset() = 0;

    virtual void
    record_position(tape &t) const = 0;

    virtual void
    reload_position(tape &t) = 0;

    virtual bool
    is_infinite() const noexcept = 0;
};

}  // namespace fairseq2n
