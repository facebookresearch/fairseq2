// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "fairseq2/native/memory.h"
#include "fairseq2/native/span.h"
#include "fairseq2/native/data/data_source.h"
#include "fairseq2/native/data/text/text.h"
#include "fairseq2/native/data/text/text_line_reader.h"

namespace fairseq2::detail {

class text_data_source final : public data_source {
public:
    explicit
    text_data_source(std::string &&pathname, text_options &&opts);

    std::optional<data>
    next() override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

private:
    std::unique_ptr<text_line_reader>
    make_text_line_reader();

    memory_block
    next_line();

    bool
    is_empty(memory_span line) const noexcept;

    [[noreturn]] void
    handle_error();

    [[noreturn]] void
    throw_read_failure();

private:
    std::string pathname_;
    text_options opts_;
    std::unique_ptr<text_line_reader> reader_;
    std::size_t num_lines_read_ = 0;
};

}  // namespace fairseq2::detail
