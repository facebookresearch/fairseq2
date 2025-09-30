// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/text/text_data_source.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <system_error>
#include <utility>

#include "fairseq2n/exception.h"
#include "fairseq2n/data/byte_stream.h"
#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/file.h"
#include "fairseq2n/data/immutable_string.h"
#include "fairseq2n/detail/exception.h"
#include "fairseq2n/utils/string.h"

namespace fairseq2n::detail {

text_data_source::text_data_source(
    std::filesystem::path &&path, std::optional<std::string> &&maybe_key, text_options &&opts)
  : path_{std::move(path)}, maybe_key_{std::move(maybe_key)}, opts_{std::move(opts)}
{
    try {
        line_reader_ = make_text_line_reader();
    } catch (const std::exception &) {
        handle_error();
    }
}

std::optional<data>
text_data_source::next()
{
    memory_block line{};

    try {
        line = read_next_line();
    } catch (const std::exception &) {
        handle_error();
    }

    if (line.empty())
        return std::nullopt;

    immutable_string output{std::move(line)};

    if (opts_.ltrim())
        output = ltrim(output);

    if (opts_.rtrim())
        output = rtrim(output);

    if (maybe_key_)
        return data_dict{{*maybe_key_, std::move(output)}};

    return output;
}

void
text_data_source::reset(bool)
{
    try {
        line_reader_->reset();
    } catch (const std::exception &) {
        handle_error();
    }
}

void
text_data_source::record_position(tape &t, bool) const
{
    line_reader_->record_position(t);
}

void
text_data_source::reload_position(tape &t, bool)
{
    line_reader_->reload_position(t);
}

data_source_finitude_type
text_data_source::finitude_type() const noexcept
{
    return data_source_finitude_type::finite;
}

std::unique_ptr<text_line_reader>
text_data_source::make_text_line_reader()
{
    constexpr std::size_t min_chunk_size = 0x0400; // 1 KiB

    std::size_t chunk_size = opts_.maybe_block_size().value_or(0x0800'0000); // 128 MiB

    auto opts = text_file_options(opts_.maybe_encoding())
        .memory_map(opts_.memory_map()).maybe_block_size(std::max(chunk_size, min_chunk_size));

    std::unique_ptr<byte_stream> stream = open_file(path_, opts);

    return std::make_unique<text_line_reader>(std::move(stream), opts_.line_ending());
}

memory_block
text_data_source::read_next_line()
{
    memory_block line{};

    while (!(line = line_reader_->next()).empty())
        if (!opts_.skip_empty() || !is_empty(line))
            break;

    return line;
}

bool
text_data_source::is_empty(memory_span line) const
{
    switch (line_reader_->actual_line_ending()) {
    case line_ending::lf:
        return line.size() == 1;
    case line_ending::crlf:
        return line.size() == 2;
    case line_ending::infer:
        throw_<internal_error>(
            "`text_data_source` has not set the line ending. Please file a bug report.");
    }

    return false;
}

void
text_data_source::handle_error()
{
    try {
        throw;
    } catch (const byte_stream_error &) {
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
    throw_with_nested<data_pipeline_error>(
        "The data pipeline cannot read from '{}'. See nested exception for details.", path_.string());
}

}  // namespace fairseq2n::detail
