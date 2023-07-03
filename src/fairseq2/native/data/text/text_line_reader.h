// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <utility>

#include "fairseq2/native/span.h"
#include "fairseq2/native/data/record_reader.h"
#include "fairseq2/native/data/stream.h"
#include "fairseq2/native/data/text/text_reader.h"

namespace fairseq2::detail {

class text_line_reader final : public record_reader {
public:
    explicit
    text_line_reader(std::unique_ptr<stream> &&s, line_ending le)
      : record_reader{std::move(s)}, line_ending_{le}
    {}

    line_ending
    actual_line_ending() const noexcept
    {
        return line_ending_;
    }

private:
    std::optional<std::size_t>
    find_record_end(memory_span chunk, bool first_chunk) override;

    bool
    infer_line_ending(span<const char> chrs);

private:
    line_ending line_ending_;
};

}  // namespace fairseq2::detail
