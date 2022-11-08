// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/utils/text.h"

#include <cstdint>

namespace fairseq2 {

invalid_utf8::~invalid_utf8() = default;

namespace detail {

std::size_t
get_code_point_length(std::string_view s)
{
    std::size_t len = 0;

    for (auto pos = s.begin(); pos < s.end(); len++) {
        std::size_t dist = 0;

        auto lead = static_cast<std::uint8_t>(*pos);
             if ((lead & 0x80) == 0x00)  // NOLINT
            dist = 1;
        else if ((lead & 0xe0) == 0xc0)  // NOLINT
            dist = 2;
        else if ((lead & 0xf0) == 0xe0)  // NOLINT
            dist = 3;
        else if ((lead & 0xf8) == 0xf0)  // NOLINT
            dist = 4;

        if (dist == 0 || static_cast<std::size_t>(s.end() - pos) < dist)
            throw invalid_utf8{"The string has an invalid UTF-8 code point."};

        pos += dist;
    }

    return len;
}

}  // namespace detail
}  // namespace fairseq2
