// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/iterator_data_source.h"
#include <iostream>
#include <execinfo.h>
#include <cstddef>

namespace fairseq2n::detail {

std::optional<data>
iterator_data_source::next()
{
    pybind11::gil_scoped_acquire acquire;
    if (iterator_ == pybind11::iterator::sentinel()) {
        return std::nullopt;
    }
    auto result = iterator_->cast<py_object>();
    ++iterator_;
    return result;
}

void
iterator_data_source::reset(bool)
{
    pybind11::gil_scoped_acquire acquire;
    iterator_.attr("reset")();
    iterator_++;
}

void
iterator_data_source::record_position(tape &t, bool) const
{
    pybind11::gil_scoped_acquire acquire;
    t.record(pybind11::module::import("pickle").attr("dumps")(iterator_).cast<py_object>());
    //t.record(iterator_.attr("__getstate__")().cast<py_object>());
}

void
iterator_data_source::reload_position(tape &t, bool)
{
    pybind11::gil_scoped_acquire acquire;
    iterator_ = pybind11::module::import("pickle").attr("loads")(pybind11::cast(t.read<py_object>()));
    //iterator_.attr("__setstate__")(pybind11::cast(t.read<py_object>()));
}

data_source_finitude_type
iterator_data_source::finitude_type() const noexcept
{
    return data_source_finitude_type::finite;
}

/*
iterator_data_source::iterator_data_source(const iterator_data_source& other) {
    pybind11::gil_scoped_acquire acquire;
    iterator_ = other.iterator_;
}

iterator_data_source& 
iterator_data_source::operator=(const iterator_data_source& other) {
    pybind11::gil_scoped_acquire acquire;
    iterator_ = other.iterator_;
    return *this;
}

iterator_data_source::~iterator_data_source()
{
    std::cout << "iterator called!\n";
    pybind11::gil_scoped_acquire acquire;
    iterator_.~iterator();
    std::cout << "iterator still ok!\n";
    void *father[10];
    size_t size;
    size = backtrace(father, 10);
    backtrace_symbols_fd(father, size, STDERR_FILENO);
}
*/

}
