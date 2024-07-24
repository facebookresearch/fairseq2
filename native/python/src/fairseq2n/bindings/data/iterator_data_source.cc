// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/bindings/data/iterator_data_source.h"

namespace py = pybind11;

namespace fairseq2n::detail {

std::optional<data>
iterator_data_source::next()
{
    py::gil_scoped_acquire acquire;

    if (reset_) {

        reloaded_ = false;
        reset_ = false;
        *iterator_ = reset_fn_(*iterator_);
        ++*iterator_;

    } else if (reloaded_) {
        // Saving/reloading the iterator may skip over an example,
        // so we check if this iterator has been reloaded and 
        // return the potentially missing example here.

        if (to_return_) {
            reloaded_ = false;
        }
        return to_return_;

    }

    if (*iterator_ == py::iterator::sentinel()) {
        return std::nullopt;
    }
    return (*iterator_)++->cast<py_object>();
}

void
iterator_data_source::reset(bool) noexcept
{
    reset_ = true;
}

void
iterator_data_source::record_position(tape &t, bool) const
{
    py::gil_scoped_acquire acquire;

    std::optional<data> to_return;
    if (*iterator_ != py::iterator::sentinel()) {
        to_return = (*iterator_)->cast<py_object>();
    }

    py::function pickle_dump_fn = py::module::import("pickle").attr("dumps");
    t.record(pickle_dump_fn(*iterator_).cast<py_object>());

    t.record(to_return);

    t.record(reset_);
}

void
iterator_data_source::reload_position(tape &t, bool)
{
    py::gil_scoped_acquire acquire;

    py::function pickle_load_fn = py::module::import("pickle").attr("loads");
    const auto& pickled_iterator = py::cast(t.read<py_object>());
    *iterator_ = pickle_load_fn(pickled_iterator);

    to_return_ = t.read<std::optional<data>>();

    reset_ = t.read<bool>();

    reloaded_ = true;
}

data_source_finitude_type
iterator_data_source::finitude_type() const noexcept
{
    if (infinite_)
        return data_source_finitude_type::infinite;
    else
        return data_source_finitude_type::finite;
}

}
