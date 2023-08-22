// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <utility>

#include "fairseq2n/api.h"

namespace fairseq2n {

class FAIRSEQ2_API py_object {
public:
    py_object() noexcept = default;

    explicit
    py_object(void *ptr, bool steal = false) noexcept
      : ptr_{ptr}
    {
        if (!steal)
            xinc_ref();
    }

    py_object(const py_object &other) noexcept
      : ptr_{other.ptr_}
    {
        xinc_ref();
    }

    py_object &
    operator=(const py_object &other) noexcept
    {
        if (this != &other) {
            xdec_ref();

            ptr_ = other.ptr_;

            xinc_ref();
        }

        return *this;
    }

    py_object(py_object &&other) noexcept
      : ptr_{other.ptr_}
    {
        other.ptr_ = nullptr;
    }

    py_object &
    operator=(py_object &&other) noexcept
    {
        if (this != &other) {
            xdec_ref();

            ptr_ = std::exchange(other.ptr_, nullptr);
        }

        return *this;
    }

   ~py_object()
    {
        xdec_ref();
    }

    void *
    release() noexcept
    {
        return std::exchange(ptr_, nullptr);
    }

    void *
    ptr() const noexcept
    {
        return ptr_;
    }

private:
    void
    xinc_ref() noexcept
    {
        if (ptr_ != nullptr)
            inc_ref();
    }

    void
    xdec_ref() noexcept
    {
        if (ptr_ != nullptr)
            dec_ref();
    }

    void
    inc_ref() noexcept;

    void
    dec_ref() noexcept;

private:
    void *ptr_{};
};

namespace detail {

FAIRSEQ2_API void
register_py_interpreter(
    void (*inc_ref_fn)(py_object &) noexcept,
    void (*dec_ref_fn)(py_object &) noexcept);

}  // namespace detail
}  // namespace fairseq2n
