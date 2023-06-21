// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/processors/str_to_tensor_converter.h"

#include <charconv>
#include <stdexcept>

#include <ATen/Dispatch.h>
#include <ATen/Functions.h>

#include <fmt/core.h>

#include "fairseq2/native/exception.h"
#include "fairseq2/native/data/immutable_string.h"

namespace fairseq2 {
namespace detail {
namespace {

const char *
get_dtype_name(at::ScalarType type)
{
    switch (type) {
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

str_to_tensor_converter::str_to_tensor_converter(std::optional<std::vector<std::int64_t>> size, std::optional<at::ScalarType> dtype)
    : size_{std::move(size)}, dtype_{dtype.value_or(at::kInt)}
{
    if (!isIntegralType(dtype_, /*includeBool=*/false))
        throw not_supported_error{"Only integral types are supported."};
}

data
str_to_tensor_converter::operator()(data &&d) const
{
    if (!d.is_string())
        throw std::invalid_argument{"The input data must be of type string."};

    std::vector<immutable_string> values = d.as_string().split(' ');

    auto dim = static_cast<std::int64_t>(values.size());

    at::Tensor tensor = at::empty(dim, dtype_);

    AT_DISPATCH_INTEGRAL_TYPES(dtype_, "fill_storage", [&] {
        fill_storage<scalar_t>(tensor, values);
    });

    if (size_)
        tensor = tensor.view(*size_);

    return tensor;
}

template <typename T>
void
str_to_tensor_converter::fill_storage(at::Tensor &t, const std::vector<immutable_string> &values) const
{
    std::int64_t idx = 0;

    auto accessor = t.accessor<T, 1>();

    for (const immutable_string &value : values) {
        const char *end = value.data() + value.size();

        T parsed_value{};

        std::from_chars_result r = std::from_chars(value.data(), end, parsed_value);
        if (r.ec == std::errc{} && r.ptr == end) {
            accessor[idx++] = parsed_value;
        } else {
            auto type_name = detail::get_dtype_name<T>();

            if (r.ec == std::errc::result_out_of_range)
                throw std::invalid_argument{
                    fmt::format("The input string must be a space-separated list of type `{0}`, but contains an element with value '{1}' that is out of range for `{0}`.", type_name, value)};
            else
                throw std::invalid_argument{
                    fmt::format("The input string must be a space-separated list of type `{0}`, but contains an element with value '{1}' that cannot be parsed as `{0}`.", type_name, value)};
        }
    }
}

}  // namespace fairseq2
