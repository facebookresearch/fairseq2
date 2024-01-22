// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#include "fairseq2n/float.h"
#include "fairseq2n/utils/cast.h"

namespace fairseq2n {
namespace detail {

template <typename T>
inline constexpr bool is_vector_v = false;

template <typename U>
inline constexpr bool is_vector_v<std::vector<U>> = true;

template <typename T>
inline constexpr bool is_optional_v = false;

template <typename U>
inline constexpr bool is_optional_v<std::optional<U>> = true;

// Used as a workaround for static_assert(false).
struct not_same {};

}  // namespace detail

template <typename T>
inline void
tape::record(const T &value)
{
    // Treat `bool` specially to avoid ambiguity with integer types.
    if constexpr (std::is_same_v<T, bool>)
        record_data(value);

    // Convert all integer types to 64-bit.
    else if constexpr (std::is_integral_v<T>)
        record_data(detail::conditional_cast<std::int64_t>(value));

    // Convert all floating-point types to double precision.
    else if constexpr (std::is_floating_point_v<T>)
        record_data(detail::conditional_cast<float64>(value));

    // Otherwise, only allow types that are implicitly convertible to `data`.
    else if constexpr (std::is_convertible_v<T, data>)
        record_data(detail::conditional_cast<data>(value));
    else
        static_assert(std::is_same_v<T, detail::not_same>,
            "T is an unsupported data type.");
}

template <typename T>
inline void
tape::record(const std::vector<T> &value)
{
    if constexpr (std::is_same_v<T, data>) {
        record_data(value);
    } else {
        record(value.size());

        for (const T &element : value)
            record(element);
    }
}

template <typename T>
inline void
tape::record(const std::optional<T> &maybe_value)
{
    if (maybe_value) {
        record_data(true);

        record(maybe_value.value());
    } else
        record_data(false);
}

template <typename T>
T
tape::read()
{
    data d = read_data();

    if constexpr (std::is_same_v<T, data>) {
        return d;
    } else if constexpr (std::is_same_v<T, bool>) {
        if (d.is_bool())
            return d.as_bool();
    } else if constexpr (std::is_integral_v<T>) {
        if (d.is_int())
            if (T i{}; detail::maybe_narrow(d.as_int(), i))
                return i;
    } else if constexpr (std::is_floating_point_v<T>) {
        if (d.is_float())
            if (T f{}; detail::maybe_narrow(d.as_float(), f))
                return f;
    } else if constexpr (std::is_same_v<T, immutable_string>) {
        if (d.is_string())
            return d.as_string();
    } else if constexpr (std::is_same_v<T, at::Tensor>) {
        if (d.is_tensor())
            return d.as_tensor();
    } else if constexpr (std::is_same_v<T, memory_block>) {
        if (d.is_memory_block())
            return d.as_memory_block();
    } else if constexpr (std::is_same_v<T, data_list>) {
        if (d.is_list())
            return d.as_list();
    } else if constexpr (std::is_same_v<T, data_dict>) {
        if (d.is_dict())
            return d.as_dict();
    } else if constexpr (std::is_same_v<T, py_object>) {
        if (d.is_py())
            return d.as_py();
    } else if constexpr (detail::is_vector_v<T>) {
        if (d.is_int()) {
            if (std::size_t size = 0; detail::maybe_narrow(d.as_int(), size)) {
                using U = typename T::value_type;

                T output{};

                output.reserve(size);

                for (std::size_t i = 0; i < size; ++i)
                    output.push_back(read<U>());

                return output;
            }
        }
    } else if constexpr (detail::is_optional_v<T>) {
        if (d.is_bool()) {
            if (d.as_bool())
                return read<typename T::value_type>();

            return std::nullopt;
        }
    } else
        static_assert(std::is_same_v<T, detail::not_same>,
            "`T` is an unsupported data type.");

    throw_corrupt();
}

}  // namespace fairseq2n
