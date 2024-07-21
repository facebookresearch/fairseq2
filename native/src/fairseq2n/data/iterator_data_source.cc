// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/iterator_data_source.h"

#include "fairseq2n/data/py.h"
#include "../python/src/fairseq2n/bindings/type_casters/py.h"

namespace fairseq2n::detail {

std::optional<data>
iterator_data_source::next()
{
    pybind11::gil_scoped_acquire acquire;
    if (reloaded_) {
        if (to_return_) {
            reloaded_ = false;
        }
        return to_return_;
    }
    if (*iterator_ == pybind11::iterator::sentinel()) {
        return std::nullopt;
    }
    return (*iterator_)++->cast<py_object>();
}

void
iterator_data_source::reset(bool)
{
    pybind11::gil_scoped_acquire acquire;

    reloaded_ = false;
    reset_fn_(*iterator_);
    ++*iterator_;
}

void
iterator_data_source::record_position(tape &t, bool) const
{
    pybind11::gil_scoped_acquire acquire;

    t.record(
        pybind11::module::import("pickle").attr("dumps")(
            *iterator_).cast<py_object>());
    std::optional<data> to_return;

    if (*iterator_ != pybind11::iterator::sentinel()) {
        to_return = (*iterator_)->cast<py_object>();
    }

    t.record(to_return);
}

void
iterator_data_source::reload_position(tape &t, bool)
{
    pybind11::gil_scoped_acquire acquire;

    *iterator_ = pybind11::module::import("pickle").attr("loads")(
        pybind11::cast(t.read<py_object>()));

    to_return_ = t.read<std::optional<data>>();

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
