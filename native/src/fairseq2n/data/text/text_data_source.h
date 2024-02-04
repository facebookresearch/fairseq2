// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
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
    text_data_source(
        std::filesystem::path &&path, std::optional<std::string> &&maybe_key, text_options &&opts);

    std::optional<data>
    next() override;

    void
    reset(bool reset_rng) override;

    void
    record_position(tape &t, bool strict) const override;

    void
    reload_position(tape &t, bool strict) override;

    data_source_finitude_type
    finitude_type() const noexcept override;

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
    std::filesystem::path path_;
    std::optional<std::string> maybe_key_;
    text_options opts_;
    std::unique_ptr<text_line_reader> line_reader_;
};

}  // namespace fairseq2n::detail
