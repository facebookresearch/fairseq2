// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>

#include <fairseq2n/data/data.h>

namespace pybind11::detail {

template <>
struct type_caster<fairseq2n::data> {
    PYBIND11_TYPE_CASTER(fairseq2n::data, const_name("Any"));

public:
    bool
    load(handle src, bool)
    {
        value = cast_from_py(src);

        return true;
    }

    static handle
    cast(const fairseq2n::data &src, return_value_policy, handle)
    {
        object obj = cast_from_cc(src);

        return obj.release();
    }

    static handle
    cast(fairseq2n::data &&src, return_value_policy, handle)
    {
        object obj = cast_from_cc(std::move(src));

        return obj.release();
    }

private:
    static fairseq2n::data
    cast_from_py(handle src);

    template <typename T>
    static object
    cast_from_cc(T &&src);
};

extern template
object
type_caster<fairseq2n::data>::cast_from_cc(const fairseq2n::data &src);

extern template
object
type_caster<fairseq2n::data>::cast_from_cc(fairseq2n::data &&src);

}  // namespace pybind11::detail
