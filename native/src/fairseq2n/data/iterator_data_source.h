// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <utility>
#include <vector>

#include "fairseq2n/data/data_source.h"
#include "fairseq2n/data/py.h"
#include "../python/src/fairseq2n/bindings/type_casters/py.h"

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

namespace fairseq2n::detail {

class iterator_data_source final : public data_source {
public:
    explicit
    iterator_data_source(pybind11::iterator &&iterator) noexcept
      : iterator_(std::move(iterator))
    { }

    iterator_data_source(const iterator_data_source& other);

    //iterator_data_source(iterator_data_source&& other);

    iterator_data_source& operator=(const iterator_data_source& other);

    //iterator_data_source& operator=(iterator_data_source&& other);

    std::optional<data>
    next() override;

    void
    reset(bool reset_rng) override;

    void
    record_position(tape &t, bool strict) const override;

    void
    reload_position(tape &t, bool strict) override;

    data_source_finitude_type
    finitude_type() const noexcept override;

    //~iterator_data_source() override;

private:
    pybind11::iterator iterator_;
};

}  // namespace fairseq2n::detail
