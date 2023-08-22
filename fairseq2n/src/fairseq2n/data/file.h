// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "fairseq2n/api.h"
#include "fairseq2n/memory.h"

namespace fairseq2n {

class byte_stream;

enum class file_mode {
    binary,
    text
};

class file_options {
public:
    file_options
    mode(file_mode value) && noexcept
    {
        mode_ = value;

        return std::move(*this);
    }

    file_mode
    mode() const noexcept
    {
        return mode_;
    }

    file_options
    memory_map(bool value) && noexcept
    {
        memory_map_ = value;

        return std::move(*this);
    }

    bool
    memory_map() const noexcept
    {
        return memory_map_;
    }

    file_options
    maybe_block_size(std::optional<std::size_t> value) && noexcept
    {
        maybe_block_size_ = value;

        return std::move(*this);
    }

    std::optional<std::size_t>
    maybe_block_size() const noexcept
    {
        return maybe_block_size_;
    }

    file_options
    maybe_text_encoding(std::optional<std::string> value) && noexcept
    {
        maybe_text_encoding_ = std::move(value);

        return std::move(*this);
    }

    const std::optional<std::string> &
    maybe_text_encoding() const noexcept
    {
        return maybe_text_encoding_;
    }

private:
    file_mode mode_ = file_mode::binary;
    bool memory_map_ = false;
    std::optional<std::size_t> maybe_block_size_{};
    std::optional<std::string> maybe_text_encoding_{};
};

inline file_options
text_file_options(std::optional<std::string> maybe_text_encoding = {}) noexcept
{
    return file_options().mode(file_mode::text).maybe_text_encoding(std::move(maybe_text_encoding));
}

FAIRSEQ2_API std::unique_ptr<byte_stream>
open_file(const std::string &pathname, const file_options &opts = {});

FAIRSEQ2_API memory_block
memory_map_file(const std::string &pathname, bool hint_sequential = false);

}  // namespace fairseq2n
