// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/py.h"

#include <cassert>

using namespace fairseq2n::detail;

namespace fairseq2n {
namespace detail {
namespace {

void (*inc_ref_fn_)(py_object &) noexcept = nullptr;
void (*dec_ref_fn_)(py_object &) noexcept = nullptr;

}  // namespace

void
register_py_interpreter(
    void (*inc_ref_fn)(py_object &) noexcept,
    void (*dec_ref_fn)(py_object &) noexcept)
{
    inc_ref_fn_ = inc_ref_fn;
    dec_ref_fn_ = dec_ref_fn;
}

}  // namespace detail

void
py_object::inc_ref() noexcept
{
    assert(inc_ref_fn_ != nullptr);

    inc_ref_fn_(*this);
}

void
py_object::dec_ref() noexcept
{
    assert(dec_ref_fn_ != nullptr);

    dec_ref_fn_(*this);
}

}  // namespace fairseq2n
