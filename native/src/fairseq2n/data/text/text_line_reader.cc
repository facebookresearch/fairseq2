// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/text/text_line_reader.h"

#include "fairseq2n/exception.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n::detail {

std::optional<std::size_t>
text_line_reader::maybe_find_record_end(memory_span chunk, bool)
{
    auto chars = cast<const char>(chunk);

    if (line_ending_ == line_ending::infer)
        if (!infer_line_ending(chars))
            return std::nullopt;

    auto pos = chars.begin();

    switch (line_ending_) {
    case line_ending::lf: {
        for (; pos < chars.end(); ++pos)
            if (*pos == '\n')
                break;

        break;
    }
    case line_ending::crlf: {
        bool has_cr = false;

        for (; pos < chars.end(); ++pos) {
            if (*pos == '\n') {
                if (has_cr)
                    break;
            } else
                has_cr = *pos == '\r';
        }

        break;
    }
    case line_ending::infer:
        throw_<internal_error>(
            "`text_line_reader` has not set the line ending. Please file a bug report.");
    }

    if (pos == chars.end())
        return std::nullopt;

    return static_cast<std::size_t>(pos - chars.begin() + 1);
}

bool
text_line_reader::infer_line_ending(span<const char> chars)
{
    bool has_cr = false;

    for (char chr : chars) {
        if (chr == '\n') {
            if (has_cr)
                line_ending_ = line_ending::crlf;
            else
                line_ending_ = line_ending::lf;

            break;
        }

        has_cr = chr == '\r';
    }

    return line_ending_ != line_ending::infer;
}

}  // namespace fairseq2n::detail
