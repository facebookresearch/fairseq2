// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/text/string_to_tensor_converter.h"

#include <stdexcept>

#include <ATen/Dispatch.h>
#include <ATen/Functions.h>

#include "fairseq2n/fmt.h"
#include "fairseq2n/utils/string.h"
#include "fairseq2n/exception.h"
#include "fairseq2n/data/immutable_string.h"
#include "fairseq2n/detail/exception.h"

using namespace fairseq2n::detail;

namespace fairseq2n {
namespace detail {
namespace {

const char *
get_dtype_name(at::ScalarType t)
{
    switch (t) {
    case at::ScalarType::Byte:
        return "torch.uint8";
    case at::ScalarType::Char:
        return "torch.int8";
    case at::ScalarType::Short:
        return "torch.int16";
    case at::ScalarType::Int:
        return "torch.int32";
    case at::ScalarType::Long:
        return "torch.int64";
    default:
        return "<unknown>";
    }
}

template <typename T>
inline const char *
get_dtype_name()
{
    return get_dtype_name(at::CppTypeToScalarType<T>::value);
}

}  // namepace
}  // namespace detail

string_to_tensor_converter::string_to_tensor_converter(
    std::vector<std::int64_t> size, std::optional<at::ScalarType> maybe_dtype)
  : size_{std::move(size)}, dtype_{maybe_dtype.value_or(at::kInt)}
{
    if (!isIntegralType(dtype_, /*includeBool=*/false))
        throw_<not_supported_error>("Only integral types are supported.");
}

data
string_to_tensor_converter::operator()(data &&d) const
{
    if (!d.is_string())
        throw_<std::invalid_argument>(
            "The input data must be of type `string`, but is of type `{}` instead.", d.type());

    immutable_string s = d.as_string();

    std::vector<immutable_string> strings{};

    s.split(' ', [&strings](immutable_string &&value)
    {
        if (!value.empty())
            strings.push_back(std::move(value));

        return true;
    });

    auto dim = static_cast<std::int64_t>(strings.size());

    at::Tensor tensor = at::empty(dim, dtype_);

    AT_DISPATCH_INTEGRAL_TYPES(dtype_, "fill_storage", [&] {
        fill_storage<scalar_t>(tensor, strings);
    });

    if (size_.empty())
        return tensor;

    return tensor.view(size_);
}

template <typename T>
void
string_to_tensor_converter::fill_storage(
    at::Tensor &tensor, const std::vector<immutable_string> &strings) const
{
    std::int64_t idx = 0;

    auto tensor_data = tensor.accessor<T, 1>();

    for (const immutable_string &s : strings) {
        try {
            tensor_data[idx++] = from_string<T>(s);
        } catch (const std::out_of_range &) {
            auto type_name = get_dtype_name<T>();

            throw_<std::invalid_argument>(
                "The input string must be a space-separated list representing values of type `{0}`, but contains an element with value '{1}' that is out of range for `{0}`.", type_name, s);
        } catch (const std::invalid_argument &) {
            auto type_name = get_dtype_name<T>();

            throw_<std::invalid_argument>(
                "The input string must be a space-separated list representing values of type `{0}`, but contains an element with value '{1}' that cannot be parsed as `{0}`.", type_name, s);
        }
    }
}

}  // namespace fairseq2n
