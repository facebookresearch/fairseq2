// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/extension/type_casters/data.h"

#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <pybind11/stl.h>

#include <ATen/Tensor.h>

#include <fairseq2/native/data/immutable_string.h>

#include "fairseq2/extension/type_casters/py.h"
#include "fairseq2/extension/type_casters/string.h"
#include "fairseq2/extension/type_casters/torch.h"

using namespace fairseq2;

namespace pybind11::detail {

data
type_caster<data>::cast_from_py(handle src)
{
    if (isinstance<bool_>(src))
        return src.cast<bool>();

    if (isinstance<int_>(src))
        return src.cast<std::int64_t>();

    if (isinstance<str>(src))
        return src.cast<std::string_view>();

    if (isinstance<immutable_string>(src))
        return src.cast<immutable_string>();

    if (isinstance<at::Tensor>(src))
        return src.cast<at::Tensor>();

    if (isinstance<bytes>(src))
        return src.cast<py_object>();

    if (isinstance<sequence>(src))
        return src.cast<std::vector<data>>();

    return src.cast<py_object>();
}

template <typename T>
object
type_caster<data>::cast_from_cc(T &&src)
{
    if (src.is_bool())
        return pybind11::cast(std::forward<T>(src).as_bool());

    if (src.is_int())
        return pybind11::cast(std::forward<T>(src).as_int());

    if (src.is_string())
        return pybind11::cast(std::forward<T>(src).as_string());

    if (src.is_tensor())
        return pybind11::cast(std::forward<T>(src).as_tensor());

    if (src.is_list())
        return pybind11::cast(std::forward<T>(src).as_list());

    if (src.is_py())
        return pybind11::cast(std::forward<T>(src).as_py());

    throw std::runtime_error{
        "The specified fairseq2::data instance cannot be converted to a Python object."};
}

template
object
type_caster<data>::cast_from_cc(const data &src);

template
object
type_caster<data>::cast_from_cc(data &&src);

}  // namespace pybind11::detail
