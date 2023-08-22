// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

namespace fairseq2n {
namespace detail {

template <typename T>
using element_type_t = std::remove_pointer_t<decltype(std::declval<T>().data())>;

// is_container
template <typename T, typename = void>
inline constexpr bool is_container_v = false;

template <typename T>
inline constexpr bool is_container_v<
    T, std::void_t<decltype(std::declval<T>().data()), decltype(std::declval<T>().size())>> = true;

// is_compatible_container
template <typename T, typename ElementT, typename = void>
inline constexpr bool is_compatible_container_v = false;

template <typename T, typename ElementT>
inline constexpr bool is_compatible_container_v<
    T, ElementT, std::enable_if_t<
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
        std::is_convertible_v<element_type_t<T>(*)[], ElementT(*)[]>>> = true;

}  // namespace detail

template <typename T>
class span {

    template <typename U>
    friend class span;

public:
    using element_type    = T;
    using value_type      = std::remove_cv_t<T>;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = T &;
    using const_reference = const T &;
    using pointer         = T *;
    using const_pointer   = const T *;
    using iterator        = T *;
    using const_iterator  = const T *;

public:
    constexpr
    span() noexcept = default;

    constexpr
    span(const span &other) noexcept = default;

    constexpr span &
    operator=(const span &) noexcept = default;

    constexpr
    span(pointer first, pointer last) noexcept
      : data_{first}, size_{static_cast<size_type>(last - first)}
    {}

    constexpr
    span(pointer data, size_type size) noexcept
      : data_{data}, size_{size}
    {}

    // The elements of `U` must be convertible to `element_type`.
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    template <typename U, typename = std::enable_if_t<std::is_convertible_v<U(*)[], T(*)[]>>>
    constexpr
    span(const span<U> &other) noexcept
      : data_{other.data_}, size_{other.size_}
    {}

    // The elements of `Container` must be convertible to `element_type`.
    template <
        typename Container,
        typename = std::enable_if_t<
            detail::is_container_v<Container> && detail::is_compatible_container_v<Container, T>>>
    constexpr
    span(Container &c)
      : data_{c.data()}, size_{static_cast<size_type>(c.size())}
    {}

    // The elements of `Container` must be convertible to `element_type`.
    template <
        typename Container,
        typename = std::enable_if_t<
            detail::is_container_v<Container> && detail::is_compatible_container_v<Container, T>>>
    constexpr
    span(const Container &c)
      : data_{c.data()}, size_{static_cast<size_type>(c.size())}
    {}

   ~span() = default;

    constexpr span<element_type>
    subspan(size_type offset) const noexcept
    {
        return {data_ + offset, size_ - offset};
    }

    constexpr span<element_type>
    subspan(size_type offset, size_type count) const noexcept
    {
        return {data_ + offset, count};
    }

    constexpr span<element_type>
    first(size_type count) const noexcept
    {
        return {data_, count};
    }

    constexpr span<element_type>
    last(size_type count) const noexcept
    {
        return {data_ + size_ - count, count};
    }

    constexpr reference
    operator[](size_type index) const noexcept
    {
        return data_[index];
    }

    constexpr iterator
    begin() const noexcept
    {
        return data_;
    }

    constexpr iterator
    end() const noexcept
    {
        return data_ + size_;
    }

    constexpr pointer
    data() const noexcept
    {
        return data_;
    }

    constexpr size_type
    size() const noexcept
    {
        return size_;
    }

    constexpr size_type
    size_bytes() const noexcept
    {
        return size_ * sizeof(element_type);
    }

    constexpr bool
    empty() const noexcept
    {
        return size_ == 0;
    }

private:
    pointer data_{};
    size_type size_{};
};

template <typename Container>
span(Container &) -> span<detail::element_type_t<Container>>;

template <typename T>
inline constexpr span<const std::byte>
as_bytes(span<T> s) noexcept
{
    return {reinterpret_cast<const std::byte *>(s.data()), s.size_bytes()};
}

// The element type of `s` must be non-const.
template <typename T, typename = std::enable_if_t<!std::is_const_v<T>>>
inline constexpr span<std::byte>
as_writable_bytes(span<T> s) noexcept
{
    return {reinterpret_cast<std::byte *>(s.data()), s.size_bytes()};
}

template <typename T>
inline constexpr span<T>
as_singleton_span(T &v) noexcept
{
    return {std::addressof(v), 1};
}

}  // namespace fairseq2n
