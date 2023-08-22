// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/bindings/type_casters/data.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <pybind11/stl.h>

#include <ATen/Tensor.h>

#include <fairseq2n/exception.h>
#include <fairseq2n/float.h>
#include <fairseq2n/data/immutable_string.h>
#include <fairseq2n/detail/exception.h>

#include "fairseq2n/bindings/type_casters/py.h"
#include "fairseq2n/bindings/type_casters/string.h"
#include "fairseq2n/bindings/type_casters/torch.h"

using namespace fairseq2n;
using namespace fairseq2n::detail;

namespace pybind11::detail {

template <typename Key, typename Value, typename Hash, typename Equal, typename Alloc>
struct type_caster<flat_hash_map<Key, Value, Hash, Equal, Alloc>>
    : map_caster<flat_hash_map<Key, Value, Hash, Equal, Alloc>, Key, Value>
{};

data
type_caster<data>::cast_from_py(handle src)
{
    if (isinstance<bool_>(src))
        return src.cast<bool>();

    if (isinstance<int_>(src))
        return src.cast<std::int64_t>();

    if (isinstance<float_>(src))
        return src.cast<float64>();

    if (isinstance<str>(src))
        return src.cast<std::string_view>();

    if (isinstance<immutable_string>(src))
        return src.cast<immutable_string>();

    if (isinstance<at::Tensor>(src))
        return src.cast<at::Tensor>();

    if (isinstance<memory_block>(src))
        return src.cast<memory_block>();

    if (isinstance<bytes>(src))
        return src.cast<py_object>();

    if (isinstance<dict>(src))
        return src.cast<data_dict>();

    if (isinstance<sequence>(src))
        return src.cast<data_list>();

    // path-like.
    if (hasattr(src, "__fspath__"))
        return src.cast<std::string_view>();

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

    if (src.is_float())
        return pybind11::cast(std::forward<T>(src).as_float());

    if (src.is_string())
        return pybind11::cast(std::forward<T>(src).as_string());

    if (src.is_tensor())
        return pybind11::cast(std::forward<T>(src).as_tensor());

    if (src.is_memory_block())
        return pybind11::cast(std::forward<T>(src).as_memory_block());

    if (src.is_list())
        return pybind11::cast(std::forward<T>(src).as_list());

    if (src.is_dict())
        return pybind11::cast(std::forward<T>(src).as_dict());

    if (src.is_py())
        return pybind11::cast(std::forward<T>(src).as_py());

    throw_<internal_error>(
        "The `data` instance cannot be converted to a Python object. Please file a bug report.");
}

template
object
type_caster<data>::cast_from_cc(const data &src);

template
object
type_caster<data>::cast_from_cc(data &&src);

}  // namespace pybind11::detail
