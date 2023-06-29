// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/text/utf8_stream.h"

#include <algorithm>
#include <system_error>
#include <utility>

#include <iconv.h>
#include <fmt/core.h>

#include "fairseq2/native/error.h"
#include "fairseq2/native/data/text/detail/utf.h"

namespace fairseq2::detail {

utf8_stream::utf8_stream(
    std::unique_ptr<stream> &&inner,
    std::optional<std::string> encoding,
    std::size_t chunk_size) noexcept
  : inner_{std::move(inner)}
{
    if (encoding)
        encoding_ = *encoding;

    is_utf8_ = is_utf8_encoding();

    constexpr std::size_t min_chunk_size = 1024;  // 1 KiB

    chunk_size_ = std::max(chunk_size, min_chunk_size);
}

utf8_stream::~utf8_stream()
{
    if (iconv_ != invalid_iconv_)
        ::iconv_close(iconv_);
}

memory_block
utf8_stream::read_chunk()
{
    if (eod_)
        return {};

    if (inner_chunk_.empty())
        inner_chunk_ = inner_->read_chunk();

    if (is_utf8_)
        return std::exchange(inner_chunk_, {});

    ensure_iconv_initialized();

    if (is_utf8_)
        return std::exchange(inner_chunk_, {});

    return read_chunk_core();
}

void
utf8_stream::reset()
{
    inner_->reset();

    inner_chunk_ = {};

    leftover_bits_ = {};

    eod_ = false;

    reset_iconv();
}

memory_block
utf8_stream::read_chunk_core()
{
    writable_memory_block chunk = allocate_memory(chunk_size_);

    writable_memory_span s = chunk;

    while (!s.empty()) {
        if (inner_chunk_.empty()) {
            if (!load_next_inner_chunk())
                break;

            move_leftover_bits();
        }

        iconv_status st = decode_inner_chunk(s);

        if (st == iconv_status::input_too_big)
            break;

        if (st == iconv_status::incomplete_sequence)
            leftover_bits_ = std::exchange(inner_chunk_, {});
    }

    if (s.empty())
        return chunk;

    return chunk.share_first(chunk.size() - s.size());
}

bool
utf8_stream::load_next_inner_chunk()
{
    inner_chunk_ = inner_->read_chunk();

    if (!inner_chunk_.empty())
        return true;

    if (!leftover_bits_.empty())
        throw stream_error{
            fmt::format("The stream ends with an invalid {} byte sequence.", encoding_)};

    eod_ = true;

    return false;
}

void
utf8_stream::move_leftover_bits()
{
    if (leftover_bits_.empty())
        return;

    writable_memory_block b = allocate_memory(leftover_bits_.size() + inner_chunk_.size());

    auto iter = std::copy(leftover_bits_.begin(), leftover_bits_.end(), b.begin());

    std::copy(inner_chunk_.begin(), inner_chunk_.end(), iter);

    inner_chunk_ = b;

    leftover_bits_ = {};
}

utf8_stream::iconv_status
utf8_stream::decode_inner_chunk(writable_memory_span &out)
{
    span<const char> inp_chrs = inner_chunk_.cast<const char>();

    span<char> out_chrs = cast<char>(out);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto inp_data = const_cast<char *>(inp_chrs.data());
    auto inp_left = inp_chrs.size();

    auto out_data = out_chrs.data();
    auto out_left = out_chrs.size();

    std::size_t r = ::iconv(iconv_, &inp_data, &inp_left, &out_data, &out_left);

    inner_chunk_ = inner_chunk_.share_last(inp_left);

    out = out.last(out_left);

    if (r != static_cast<std::size_t>(-1))
        return iconv_status::ok;

    std::error_code err = last_error();

    if (err == std::errc::argument_list_too_long)
        return iconv_status::input_too_big;

    if (err == std::errc::invalid_argument)
        return iconv_status::incomplete_sequence;

    if (err == std::errc::illegal_byte_sequence)
        throw stream_error{
            fmt::format("An invalid {} byte sequence encountered.", encoding_)};

    throw std::system_error{err,
        fmt::format("A system error occurred while decoding the {} stream", encoding_)};
}

void
utf8_stream::ensure_iconv_initialized()
{
    if (iconv_ != invalid_iconv_)
        return;

    if (encoding_.empty()) {
        encoding_ = infer_bom_encoding(inner_chunk_);

        if (is_utf8_encoding()) {
            is_utf8_ = true;

            return;
        }
    }

    iconv_ = ::iconv_open("UTF-8", encoding_.c_str());
    if (iconv_ != invalid_iconv_)
        return;

    std::error_code err = last_error();

    if (err == std::errc::invalid_argument)
        throw std::system_error{err,
            fmt::format("The {} encoding is not supported by the system", encoding_)};

    throw std::system_error{err};
}

inline void
utf8_stream::reset_iconv() noexcept
{
    if (iconv_ != invalid_iconv_)
        ::iconv(iconv_, nullptr, nullptr, nullptr, nullptr);
}

// NOLINTNEXTLINE(performance-no-int-to-ptr)
const ::iconv_t utf8_stream::invalid_iconv_ = reinterpret_cast<::iconv_t>(-1);

}  // namespace fairseq2::detail
