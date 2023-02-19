// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>

#include <fairseq2/native/data/data.h>

template <>
struct pybind11::detail::type_caster<fairseq2::data> {
    PYBIND11_TYPE_CASTER(fairseq2::data, pybind11::detail::const_name("Any"));

public:
    bool
    load(pybind11::handle src, bool)
    {
        value = cast_from_py(src);

        return true;
    }

    static pybind11::handle
    cast(const fairseq2::data &src, pybind11::return_value_policy, pybind11::handle)
    {
        pybind11::object o = cast_from_cc(src);

        return o.release();
    }

    static pybind11::handle
    cast(fairseq2::data &&src, pybind11::return_value_policy, pybind11::handle)
    {
        pybind11::object o = cast_from_cc(std::move(src));

        return o.release();
    }

private:
    static fairseq2::data
    cast_from_py(pybind11::handle src);

    template <typename T>
    static pybind11::object
    cast_from_cc(T &&src);
};

extern template
pybind11::object
pybind11::detail::type_caster<fairseq2::data>::cast_from_cc(const fairseq2::data &src);

extern template
pybind11::object
pybind11::detail::type_caster<fairseq2::data>::cast_from_cc(fairseq2::data &&src);
