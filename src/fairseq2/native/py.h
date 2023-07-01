// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <utility>

#include "fairseq2/native/api.h"

namespace fairseq2 {
namespace detail {

class py_gil_acquire {
public:
    py_gil_acquire() noexcept
    {
        ensure_thread_state_set();
    }

    py_gil_acquire(const py_gil_acquire &) = delete;
    py_gil_acquire &operator=(const py_gil_acquire &) = delete;

    py_gil_acquire(py_gil_acquire &&) = delete;
    py_gil_acquire &operator=(py_gil_acquire &&) = delete;

   ~py_gil_acquire()
    {
        release_gil();
    }

private:
   void
   ensure_thread_state_set() noexcept;

   void
   release_gil() const noexcept;

private:
    bool release_thread_state_{};
};

class py_gil_release {
public:
    py_gil_release() noexcept
    {
        release_gil();
    }

    py_gil_release(const py_gil_release &) = delete;
    py_gil_release &operator=(const py_gil_release &) = delete;

    py_gil_release(py_gil_release &&) = delete;
    py_gil_release &operator=(py_gil_release &&) = delete;

   ~py_gil_release()
    {
        acquire_gil();
    }

private:
   void
   release_gil() noexcept;

   void
   acquire_gil() noexcept;

private:
    void *thread_state_ = nullptr;
};

FAIRSEQ2_API bool
py_is_finalizing() noexcept;

void
throw_if_py_is_finalizing();

}  // namespace detail

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
        // NOTE [Python Finalization]
        //
        // This likely has an impact on the runtime performance, but, as of
        // CPython 3.10, there is no graceful way to dispose Python objects
        // used in background threads.
        //
        // Also note that there is still a race condition between this check
        // and the time we attempt to acquire GIL. So this is only a partial
        // mitigation.
        //
        // See https://github.com/python/cpython/pull/28525.
        if (detail::py_is_finalizing())
            return;

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

}  // namespace fairseq2
