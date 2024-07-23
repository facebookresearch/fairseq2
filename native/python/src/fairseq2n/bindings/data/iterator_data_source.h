// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/data_source.h"
#include "fairseq2n/bindings/type_casters/py.h"

#include <pybind11/pybind11.h>


namespace py = pybind11;

using reset_fn = std::function<py::iterator(py::iterator &)>;

namespace fairseq2n::detail {

class iterator_data_source final : public data_source {
public:
    explicit
    iterator_data_source(
        py::iterator &&iterator, 
        reset_fn &&fn, 
        bool infinite)
      : iterator_{new py::iterator{std::move(iterator)}},
        reset_fn_{std::move(fn)},
        infinite_{infinite}
    {
        reset(true);
    }

    std::optional<data>
    next() override;

    void
    reset(bool reset_rng) noexcept override;

    void
    record_position(tape &t, bool strict) const override;

    void
    reload_position(tape &t, bool strict) override;

    data_source_finitude_type
    finitude_type() const noexcept override;

private:
    struct iterator_deleter {
        void operator()(py::iterator* it) {
            py::gil_scoped_acquire acquire;

            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
            delete it;
        }
    };

    std::unique_ptr<py::iterator,  iterator_deleter> iterator_;
    reset_fn reset_fn_;
    bool infinite_;
    std::optional<data> to_return_;
    bool reloaded_{false};
    bool reset_{false};
};

}  // namespace fairseq2n::detail
