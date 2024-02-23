// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>

#include <fairseq2n/data/immutable_string.h>

namespace pybind11::detail {

template <>
struct type_caster<fairseq2n::immutable_string> : string_caster<fairseq2n::immutable_string>
{};

}  // namespace pybind11::detail
