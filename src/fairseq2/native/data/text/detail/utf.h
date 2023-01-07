// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

#include "fairseq2/native/api.h"
#include "fairseq2/native/memory.h"

namespace fairseq2 {
namespace detail {

std::size_t
compute_code_point_length(std::string_view s);

std::string
infer_bom_encoding(memory_span preamble) noexcept;

}

class FAIRSEQ2_API invalid_utf8_error : public std::logic_error {
public:
    using std::logic_error::logic_error;

public:
    invalid_utf8_error(const invalid_utf8_error &) = default;
    invalid_utf8_error &operator=(const invalid_utf8_error &) = default;

   ~invalid_utf8_error() override;
};

}  // namespace fairseq2
