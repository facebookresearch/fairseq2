// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

#include "fairseq2/native/api.h"
#include "fairseq2/native/utils/span.h"

namespace fairseq2::detail {

using memory_deallocator = void (*)(const void *ptr, std::size_t size) noexcept;

class FAIRSEQ2_API memory_holder {
public:
    explicit constexpr
    memory_holder(const void *ptr, std::size_t size, memory_deallocator d) noexcept
        : ptr_{ptr}, size_{size}, deallocate_{d}
    {}

    memory_holder(const memory_holder &) = delete;
    memory_holder &operator=(const memory_holder &) = delete;

    memory_holder(memory_holder &&) = delete;
    memory_holder &operator=(memory_holder &&) = delete;

    ~memory_holder()
    {
        if (ptr_ != nullptr && deallocate_ != nullptr)
            deallocate_(ptr_, size_);
    }

private:
    const void *ptr_;
    std::size_t size_;
    memory_deallocator deallocate_;
};

// A `memory_block` is intended to be used for zero-copy memory sharing. The
// underlying memory is reference-counted and is freed when no longer needed.
template <typename T>
class basic_memory_block {

    template <typename U>
    friend class basic_memory_block;

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

    constexpr
    basic_memory_block() noexcept = default;

    explicit constexpr
    basic_memory_block(pointer data, size_type size) noexcept
        : data_{data}, size_{size}
    {}

    explicit
    basic_memory_block(pointer data, size_type size, memory_deallocator d);

    template <typename U>
    basic_memory_block(const basic_memory_block<U> &other) noexcept
        : data_{other.data_}, size_{other.size_}, holder_{other.holder_}
    {
        static_assert(std::is_convertible_v<U *, T *>,
            "A `memory_block` cannot be converted into a `mutable_memory_block`.");
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

    constexpr bool
    empty() const noexcept
    {
        return size_ == 0;
    }

    template <typename U>
    constexpr span<U>
    cast() const noexcept
    {
        static_assert(std::is_const_v<U> || !std::is_const_v<T>,
            "A `memory_block` cannot be casted to a non-const type.");

        return {reinterpret_cast<U *>(data_), size_ / sizeof(U)};
    }

private:
    pointer data_{};
    size_type size_{};
    std::shared_ptr<memory_holder> holder_{};
};

template <typename T>
basic_memory_block<T>::basic_memory_block(pointer data, size_type size, memory_deallocator d)
    : data_{data}, size_{size}
{
    // As a contract, we take the ownership of the provided pointer within the
    // constructor. This means we have to make sure that we don't leak in case
    // of a failure.
    try {
        holder_ = std::make_shared<memory_holder>(data, size, d);
    } catch (...) {
        if (data != nullptr && d != nullptr)
            d(data, size);

        throw;
    }
}

// Actual template specializations
using memory_block = basic_memory_block<const std::byte>;
using mutable_memory_block = basic_memory_block<std::byte>;

// Type aliases for non-owning memory references
using memory_span = span<const std::byte>;
using mutable_memory_span = span<std::byte>;

mutable_memory_block
allocate_host_memory(std::size_t size);

}  // namespace fairseq2::detail
