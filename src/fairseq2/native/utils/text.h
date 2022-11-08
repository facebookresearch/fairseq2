// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

#include "fairseq2/native/api.h"

namespace fairseq2 {

class FAIRSEQ2_API invalid_utf8 : public std::logic_error {
public:
    using std::logic_error::logic_error;

    invalid_utf8(const invalid_utf8 &) = default;
    invalid_utf8 &operator=(const invalid_utf8 &) = default;

    ~invalid_utf8() override;
};

namespace detail {

std::size_t
get_code_point_length(std::string_view s);

}  // namespace detail
}  // namespace fairseq2
