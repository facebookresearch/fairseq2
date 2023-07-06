// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <vector>
#include <optional>
#include <type_traits>

#include "fairseq2/native/float.h"
#include "fairseq2/native/utils/cast.h"

namespace fairseq2 {
namespace detail {

template <typename T>
inline constexpr bool is_vector_v = false;

template <typename T>
inline constexpr bool is_vector_v<std::vector<T>> = true;

template <typename T>
inline constexpr bool is_optional_v = false;

template <typename T>
inline constexpr bool is_optional_v<std::optional<T>> = true;

// Used as a workaround for static_assert(false).
struct not_same {};

}  // namespace detail

template <typename T>
inline void
tape::record(const T &d)
{
    if constexpr (std::is_same_v<T, bool>)
        record_data(d);
    else if constexpr (std::is_integral_v<T>)
        record_data(detail::conditional_cast<std::int64_t>(d));
    else if constexpr (std::is_floating_point_v<T>)
        record_data(detail::conditional_cast<float64>(d));
    else if constexpr (std::is_convertible_v<T, data>)
        record_data(detail::conditional_cast<data>(d));
    else
        static_assert(std::is_same_v<T, detail::not_same>,
            "T is an unsupported data type.");
}

template <typename T>
inline void
tape::record(const std::vector<T> &d)
{
    if constexpr (std::is_same_v<T, data>) {
        record_data(d);
    } else {
        record(d.size());

        for (const T &v : d)
            record(v);
    }
}

template <typename T>
inline void
tape::record(const std::optional<T> &d)
{
    if (d) {
        record_data(true);

        record(*d);
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
            if (T i{}; detail::try_narrow(d.as_int(), i))
                return i;
    } else if constexpr (std::is_floating_point_v<T>) {
        if (d.is_float())
            if (T f{}; detail::try_narrow(d.as_float(), f))
                return f;
    } else if constexpr (std::is_same_v<T, immutable_string>) {
        if (d.is_string())
            return d.as_string();
    } else if constexpr (std::is_same_v<T, at::Tensor>) {
        if (d.is_tensor())
            return d.as_tensor();
    } else if constexpr (std::is_same_v<T, std::vector<data>>) {
        if (d.is_list())
            return d.as_list();
    } else if constexpr (std::is_same_v<T, py_object>) {
        if (d.is_py())
            return d.as_py();
    } else if constexpr (detail::is_vector_v<T>) {
        if (d.is_int()) {
            if (std::size_t size = 0; detail::try_narrow(d.as_int(), size)) {
                using U = typename T::value_type;

                std::vector<U> output{};

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

}  // namespace fairseq2
