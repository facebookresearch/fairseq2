// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/data_source.h"

#include <pybind11/pybind11.h>

namespace fairseq2n::detail {

class iterator_data_source final : public data_source {
public:
    explicit
    iterator_data_source(
        pybind11::iterator &&iterator, 
        reset_fn &&fn, 
        bool infinite)
      : iterator_{new pybind11::iterator{std::move(iterator)}},
        reset_fn_{std::move(fn)},
        infinite_{infinite}
    {
        reset(true);
    }

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

private:
    struct iterator_deleter {
        void operator()(pybind11::iterator* it) {
            pybind11::gil_scoped_acquire acquire;

            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
            delete it;
        }
    };

    std::unique_ptr<pybind11::iterator,  iterator_deleter> iterator_;
    reset_fn reset_fn_;
    bool infinite_;
    std::optional<data> to_return_;
    bool reloaded_{false};
};

}  // namespace fairseq2n::detail
