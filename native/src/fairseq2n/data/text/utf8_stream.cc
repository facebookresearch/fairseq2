// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/text/utf8_stream.h"

#include <algorithm>
#include <utility>
#include <system_error>

#include <iconv.h>

#include "fairseq2n/data/text/detail/utf.h"
#include "fairseq2n/detail/error.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n::detail {

utf8_stream::utf8_stream(
    std::unique_ptr<byte_stream> &&inner,
    std::optional<std::string> maybe_encoding,
    std::size_t chunk_size) noexcept
  : inner_{std::move(inner)}
{
    if (maybe_encoding)
        encoding_ = *std::move(maybe_encoding);

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
    if (is_eod_)
        return {};

    if (encoded_chunk_.empty())
        encoded_chunk_ = inner_->read_chunk();

    // If the stream is already in UTF-8, take the shortcut.
    if (is_utf8_)
        return std::exchange(encoded_chunk_, {});

    ensure_iconv_initialized();

    // If iconv is initialized in this call, `is_utf8_` might be updated. Let's
    // double check it.
    if (is_utf8_)
        return std::exchange(encoded_chunk_, {});

    return do_read_chunk();
}

void
utf8_stream::reset()
{
    inner_->reset();

    encoded_chunk_ = {};

    leftover_bits_ = {};

    is_eod_ = false;

    reset_iconv();
}

memory_block
utf8_stream::do_read_chunk()
{
    writable_memory_block decoded_chunk = allocate_memory(chunk_size_);

    writable_memory_span remaining_space = decoded_chunk;

    while (!remaining_space.empty()) {
        if (encoded_chunk_.empty()) {
            if (!load_next_encoded_chunk())
                break;

            move_leftover_bits();
        }

        iconv_status st = decode_encoded_chunk(remaining_space);

        if (st == iconv_status::input_too_big)
            break;

        if (st == iconv_status::incomplete_sequence)
            leftover_bits_ = std::exchange(encoded_chunk_, {});
    }

    if (remaining_space.empty())
        return decoded_chunk;

    return decoded_chunk.share_first(decoded_chunk.size() - remaining_space.size());
}

bool
utf8_stream::load_next_encoded_chunk()
{
    encoded_chunk_ = inner_->read_chunk();

    if (!encoded_chunk_.empty())
        return true;

    if (!leftover_bits_.empty())
        throw_<byte_stream_error>(
            "The stream ends with an invalid {} byte sequence.", encoding_);

    is_eod_ = true;

    return false;
}

void
utf8_stream::move_leftover_bits()
{
    if (leftover_bits_.empty())
        return;

    writable_memory_block block = allocate_memory(leftover_bits_.size() + encoded_chunk_.size());

    auto pos = std::copy(leftover_bits_.begin(), leftover_bits_.end(), block.begin());

    std::copy(encoded_chunk_.begin(), encoded_chunk_.end(), pos);

    encoded_chunk_ = block;

    leftover_bits_ = {};
}

utf8_stream::iconv_status
utf8_stream::decode_encoded_chunk(writable_memory_span &output)
{
    span<const char> input_chars = encoded_chunk_.cast<const char>();

    span<char> output_chars = cast<char>(output);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto input_data = const_cast<char *>(input_chars.data());
    auto input_left = input_chars.size();

    auto output_data = output_chars.data();
    auto output_left = output_chars.size();

    std::size_t result = ::iconv(iconv_, &input_data, &input_left, &output_data, &output_left);

    encoded_chunk_ = encoded_chunk_.share_last(input_left);

    output = output.last(output_left);

    if (result != static_cast<std::size_t>(-1))
        return iconv_status::ok;

    std::error_code err = last_error();

    if (err == std::errc::argument_list_too_long)
        return iconv_status::input_too_big;

    if (err == std::errc::invalid_argument)
        return iconv_status::incomplete_sequence;

    if (err == std::errc::illegal_byte_sequence)
        throw_<byte_stream_error>(
            "An invalid {} byte sequence has been encountered.", encoding_);

    throw_system_error(err,
        "The stream cannot be decoded as {}.", encoding_);
}

void
utf8_stream::ensure_iconv_initialized()
{
    if (iconv_ != invalid_iconv_)
        return;

    if (encoding_.empty()) {
        encoding_ = infer_bom_encoding(encoded_chunk_);

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
        throw_system_error(err,
            "The {} encoding is not supported by the system", encoding_);

    throw_system_error(err,
        "The stream cannot be decoded as {}.", encoding_);
}

inline void
utf8_stream::reset_iconv() noexcept
{
    if (iconv_ != invalid_iconv_)
        ::iconv(iconv_, nullptr, nullptr, nullptr, nullptr);
}

// NOLINTNEXTLINE(performance-no-int-to-ptr)
const ::iconv_t utf8_stream::invalid_iconv_ = reinterpret_cast<::iconv_t>(-1);

}  // namespace fairseq2n::detail
