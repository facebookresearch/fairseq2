// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "fairseq2/native/api.h"

namespace fairseq2 {

enum class line_ending {
    infer,
    lf,
    crlf
};

class text_options {
public:
    text_options &
    encoding(std::optional<std::string> value) & noexcept
    {
        encoding_ = std::move(value);

        return *this;
    }

    text_options &&
    encoding(std::optional<std::string> value) && noexcept
    {
        encoding_ = std::move(value);

        return std::move(*this);
    }

    const std::optional<std::string> &
    encoding() const noexcept
    {
        return encoding_;
    }

    text_options &
    line_ending(fairseq2::line_ending value) & noexcept
    {
        line_ending_ = value;

        return *this;
    }

    text_options &&
    line_ending(fairseq2::line_ending value) && noexcept
    {
        line_ending_ = value;

        return std::move(*this);
    }

    fairseq2::line_ending
    line_ending() const noexcept
    {
        return line_ending_;
    }

    text_options &
    ltrim(bool value) & noexcept
    {
        ltrim_ = value;

        return *this;
    }

    text_options &&
    ltrim(bool value) && noexcept
    {
        ltrim_ = value;

        return std::move(*this);
    }

    bool
    ltrim() const noexcept
    {
        return ltrim_;
    }

    text_options &
    rtrim(bool value) & noexcept
    {
        rtrim_ = value;

        return *this;
    }

    text_options &&
    rtrim(bool value) && noexcept
    {
        rtrim_ = value;

        return std::move(*this);
    }

    bool
    rtrim() const noexcept
    {
        return rtrim_;
    }

    text_options &
    skip_empty(bool value) & noexcept
    {
        skip_empty_ = value;

        return *this;
    }

    text_options &&
    skip_empty(bool value) && noexcept
    {
        skip_empty_ = value;

        return std::move(*this);
    }

    bool
    skip_empty() const noexcept
    {
        return skip_empty_;
    }

    text_options &
    memory_map(bool value) & noexcept
    {
        memory_map_ = value;

        return *this;
    }

    text_options &&
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

    text_options &
    block_size(std::optional<std::size_t> value) & noexcept
    {
        block_size_ = value;

        return *this;
    }

    text_options &&
    block_size(std::optional<std::size_t> value) && noexcept
    {
        block_size_ = value;

        return std::move(*this);
    }

    std::optional<std::size_t>
    block_size() const noexcept
    {
        return block_size_;
    }

private:
    std::optional<std::string> encoding_{};
    fairseq2::line_ending line_ending_{};
    bool ltrim_ = false;
    bool rtrim_ = false;
    bool skip_empty_ = false;
    bool memory_map_ = false;
    std::optional<std::size_t> block_size_{};
};

class data_pipeline_builder;

FAIRSEQ2_API data_pipeline_builder
read_text(std::string pathname, text_options opts = {});

}  // namespace fairseq2
