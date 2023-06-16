// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/text/text_data_source.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <system_error>
#include <utility>

#include <fmt/core.h>

#include "fairseq2/native/error.h"
#include "fairseq2/native/data/data_pipeline.h"
#include "fairseq2/native/data/file.h"
#include "fairseq2/native/data/immutable_string.h"
#include "fairseq2/native/data/stream.h"
#include "fairseq2/native/utils/string.h"

namespace fairseq2::detail {

text_data_source::text_data_source(std::string &&pathname, text_options &&opts)
    : pathname_{std::move(pathname)}, opts_{std::move(opts)}
{
    try {
        reader_ = make_text_line_reader();
    } catch (const std::exception &) {
        handle_error();
    }
}

std::optional<data>
text_data_source::next()
{
    memory_block line;

    try {
        line = next_line();
    } catch (const std::exception &) {
        handle_error();
    }

    if (line.empty())
        return {};

    immutable_string example{std::move(line)};

    if (opts_.ltrim())
        example = ltrim(example);

    if (opts_.rtrim())
        example = rtrim(example);

    return example;
}

void
text_data_source::reset()
{
    try {
        reader_->reset();
    } catch (const std::exception &) {
        handle_error();
    }

    num_lines_read_ = 0;
}

void
text_data_source::record_position(tape &t) const
{
    t.record(num_lines_read_);
}

void
text_data_source::reload_position(tape &t)
{
    auto num_lines_read = t.read<std::size_t>();

    reset();

    for (std::size_t i = 0; i < num_lines_read; i++)
        next_line();
}

std::unique_ptr<text_line_reader>
text_data_source::make_text_line_reader()
{
    constexpr std::size_t min_chunk_size = 0x0400; // 1 KiB

    std::size_t chunk_size = opts_.block_size().value_or(0x0800'0000); // 128 MiB

    auto opts = text_file_options(opts_.encoding())
        .memory_map(opts_.memory_map()).block_size(std::max(chunk_size, min_chunk_size));

    std::unique_ptr<stream> s = read_file(pathname_, opts);

    return std::make_unique<text_line_reader>(std::move(s), opts_.line_ending());
}

memory_block
text_data_source::next_line()
{
    memory_block line{};

    while (!(line = reader_->next()).empty()) {
        if (!opts_.skip_empty() || !is_empty(line))
            break;
    }

    num_lines_read_++;

    return line;
}

bool
text_data_source::is_empty(memory_span line) const noexcept
{
    switch (reader_->actual_line_ending()) {
    case line_ending::lf:
        return line.size() == 1;
    case line_ending::crlf:
        return line.size() == 2;
    case line_ending::infer:
        unreachable();
    }

    return false;
}

void
text_data_source::handle_error()
{
    try {
        throw;
    } catch (const stream_error &) {
        throw_read_failure();
    } catch (const record_error &) {
        throw_read_failure();
    } catch (const std::system_error &) {
        throw_read_failure();
    }
}

inline void
text_data_source::throw_read_failure()
{
    data_pipeline_error::throw_nested(
        fmt::format("The data pipeline cannot read from '{}'.", pathname_));
}

}  // namespace fairseq2::detail
