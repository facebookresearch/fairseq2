// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include "fairseq2n/memory.h"

namespace fairseq2n::detail {

std::size_t
compute_code_point_length(std::string_view s);

std::string
infer_bom_encoding(memory_span preamble) noexcept;

}  // namespace fairseq2n::detail
