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

#include "fairseq2n/memory.h"
#include "fairseq2n/span.h"
#include "fairseq2n/data/data_source.h"
#include "fairseq2n/data/text/text_line_reader.h"
#include "fairseq2n/data/text/text_reader.h"

namespace fairseq2n::detail {

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

    bool
    is_infinite() const noexcept override;

private:
    std::unique_ptr<text_line_reader>
    make_text_line_reader();

    memory_block
    read_next_line();

    bool
    is_empty(memory_span line) const;

    [[noreturn]] void
    handle_error();

    [[noreturn]] void
    throw_read_failure();

private:
    std::string pathname_;
    text_options opts_;
    std::unique_ptr<text_line_reader> line_reader_;
    std::size_t num_lines_read_ = 0;
};

}  // namespace fairseq2n::detail
