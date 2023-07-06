// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/text/string_to_tensor_converter.h"

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
    std::optional<std::vector<std::int64_t>> size, std::optional<at::ScalarType> dtype)
  : size_{std::move(size)}, dtype_{dtype.value_or(at::kInt)}
{
    if (!isIntegralType(dtype_, /*includeBool=*/false))
        throw not_supported_error{"Only integral types are supported."};
}

data
string_to_tensor_converter::operator()(data &&d) const
{
    if (!d.is_string())
        throw std::invalid_argument{"The input data must be of type string."};

    std::vector<immutable_string> strings = d.as_string().split(' ');

    auto dim = static_cast<std::int64_t>(strings.size());

    at::Tensor tensor = at::empty(dim, dtype_);

    AT_DISPATCH_INTEGRAL_TYPES(dtype_, "fill_storage", [&] {
        fill_storage<scalar_t>(tensor, strings);
    });

    if (size_)
        tensor = tensor.view(*size_);

    return tensor;
}

template <typename T>
void
string_to_tensor_converter::fill_storage(
    at::Tensor &tensor, const std::vector<immutable_string> &strings) const
{
    std::int64_t idx = 0;

    auto tensor_accessor = tensor.accessor<T, 1>();

    for (const immutable_string &s : strings) {
        const char *str_end = s.data() + s.size();

        T parsed_value{};

        std::from_chars_result result = std::from_chars(s.data(), str_end, parsed_value);
        if (result.ec == std::errc{} && result.ptr == str_end) {
            tensor_accessor[idx++] = parsed_value;
        } else {
            auto type_name = detail::get_dtype_name<T>();

            if (result.ec == std::errc::result_out_of_range)
                throw std::invalid_argument{
                    fmt::format("The input string must be a space-separated list of type `{0}`, but contains an element with value '{1}' that is out of range for `{0}`.", type_name, s)};
            else
                throw std::invalid_argument{
                    fmt::format("The input string must be a space-separated list of type `{0}`, but contains an element with value '{1}' that cannot be parsed as `{0}`.", type_name, s)};
        }
    }
}

}  // namespace fairseq2
