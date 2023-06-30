// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/py.h"

#include <cassert>
#include <stdexcept>

#include <dlfcn.h>

using namespace fairseq2::detail;

namespace {

int
Py_IsFinalizing() noexcept
{
    static auto fn = reinterpret_cast<int (*)()>(
        ::dlsym(RTLD_DEFAULT, "_Py_IsFinalizing"));

    assert(fn != nullptr);

    return fn();
}

void
Py_IncRef(void *ptr) noexcept
{
    static auto fn = reinterpret_cast<void (*)(void *)>(
        ::dlsym(RTLD_DEFAULT, "Py_IncRef"));

    assert(fn != nullptr);

    fn(ptr);
}

void
Py_DecRef(void *ptr) noexcept
{
    static auto fn = reinterpret_cast<void (*)(void *)>(
        ::dlsym(RTLD_DEFAULT, "Py_DecRef"));

    assert(fn != nullptr);

    fn(ptr);
}

int
PyGILState_Check() noexcept
{
    static auto fn = reinterpret_cast<int (*)()>(
        ::dlsym(RTLD_DEFAULT, "PyGILState_Check"));

    if (fn == nullptr)
        return 0;

    return fn();
}

enum PyGILState_STATE {PyGILState_LOCKED, PyGILState_UNLOCKED};

PyGILState_STATE
PyGILState_Ensure() noexcept
{
    static auto fn = reinterpret_cast<PyGILState_STATE (*)()>(
        ::dlsym(RTLD_DEFAULT, "PyGILState_Ensure"));

    assert(fn != nullptr);

    return fn();
}

void
PyGILState_Release(PyGILState_STATE s) noexcept
{
    static auto fn = reinterpret_cast<void (*)(PyGILState_STATE)>(
        ::dlsym(RTLD_DEFAULT, "PyGILState_Release"));

    assert(fn != nullptr);

    fn(s);
}

void *
PyEval_SaveThread() noexcept
{
    static auto fn = reinterpret_cast<void * (*)()>(
        ::dlsym(RTLD_DEFAULT, "PyEval_SaveThread"));

    assert(fn != nullptr);

    return fn();
}

void
PyEval_RestoreThread(void *tstate) noexcept
{
    static auto fn = reinterpret_cast<void (*)(void *)>(
        ::dlsym(RTLD_DEFAULT, "PyEval_RestoreThread"));

    assert(fn != nullptr);

    fn(tstate);
}

}  // namespace

namespace fairseq2 {
namespace detail {

void
py_gil_acquire::ensure_thread_state_set() noexcept
{
    release_thread_state_ = PyGILState_Ensure() == PyGILState_UNLOCKED;
}

void
py_gil_acquire::release_gil() const noexcept
{
    PyGILState_Release(release_thread_state_ ? PyGILState_UNLOCKED : PyGILState_LOCKED);
}

void
py_gil_release::release_gil() noexcept
{
    if (PyGILState_Check() == 1)
        thread_state_ = PyEval_SaveThread();
}

void
py_gil_release::acquire_gil() noexcept
{
    if (thread_state_ != nullptr)
        PyEval_RestoreThread(thread_state_);
}

bool
py_is_finalizing() noexcept
{
    return Py_IsFinalizing() != 0;
}

}  // namespace detail

void
py_object::inc_ref() noexcept
{
    py_gil_acquire gil{};

    Py_IncRef(ptr_);
}

void
py_object::dec_ref() noexcept
{
    py_gil_acquire gil{};

    Py_DecRef(ptr_);
}

}  // namespace fairseq2
