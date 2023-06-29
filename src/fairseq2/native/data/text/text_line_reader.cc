// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/text/text_line_reader.h"

#include "fairseq2/native/error.h"

namespace fairseq2::detail {

std::optional<std::size_t>
text_line_reader::find_record_end(memory_span chunk, bool)
{
    auto chrs = cast<const char>(chunk);

    if (line_ending_ == line_ending::infer)
        if (!infer_line_ending(chrs))
            return std::nullopt;

    auto iter = chrs.begin();

    switch (line_ending_) {
    case line_ending::lf: {
        for (; iter < chrs.end(); ++iter) {
            if (*iter == '\n')
                break;
        }

        break;
    }
    case line_ending::crlf: {
        bool has_cr = false;

        for (; iter < chrs.end(); ++iter) {
            if (*iter == '\n') {
                if (has_cr)
                    break;
            } else {
                has_cr = *iter == '\r';
            }
        }

        break;
    }
    case line_ending::infer:
        unreachable();
    }

    if (iter == chrs.end())
        return std::nullopt;

    return static_cast<std::size_t>(iter - chrs.begin() + 1);
}

bool
text_line_reader::infer_line_ending(span<const char> chrs)
{
    bool has_cr = false;

    for (char c : chrs) {
        if (c == '\n') {
            if (has_cr)
                line_ending_ = line_ending::crlf;
            else
                line_ending_ = line_ending::lf;

            break;
        }

        has_cr = c == '\r';
    }

    return line_ending_ != line_ending::infer;
}

}  // namespace fairseq2::detail
