// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/interop.h"

template <>
struct FAIRSEQ2_API pybind11::detail::type_caster<fairseq2::ivariant> {
public:
    PYBIND11_TYPE_CASTER(fairseq2::ivariant, pybind11::detail::const_name("IVariant"));

    bool
    load(pybind11::handle src, bool)
    {
        value = cast_from_py(src);

        return !value.is_uninitialized();
    }

    static pybind11::handle
    cast(const fairseq2::ivariant &src, pybind11::return_value_policy, pybind11::handle)
    {
        return cast_from_cc(src).inc_ref();
    }

private:
    static pybind11::object
    cast_from_cc(const fairseq2::ivariant &src);

    static fairseq2::ivariant
    cast_from_py(pybind11::handle src);
};

PYBIND11_MAKE_OPAQUE(fairseq2::ilist);
PYBIND11_MAKE_OPAQUE(fairseq2::idict);
