// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fairseq2n/api.h"

namespace fairseq2n {

class sp_model_options {
public:
    sp_model_options
    control_symbols(std::string value) &&
    {
        control_symbols_.push_back(std::move(value));

        return std::move(*this);
    }

    std::vector<std::string> &
    control_symbols() noexcept
    {
        return control_symbols_;
    }

    const std::vector<std::string> &
    control_symbols() const noexcept
    {
        return control_symbols_;
    }

private:
    std::vector<std::string> control_symbols_{};
};

namespace detail {

class sp_processor;

}

class FAIRSEQ2_API sp_model {
    friend class sp_decoder;
    friend class sp_encoder;

public:
    static sp_model
    from_serialized(std::string_view serialized);

private:
    explicit
    sp_model(std::unique_ptr<detail::sp_processor> &&processor) noexcept;

public:
    explicit
    sp_model(std::string_view pathname, sp_model_options opts = {});

    sp_model(const sp_model &) = delete;
    sp_model &operator=(const sp_model &) = delete;

    sp_model(sp_model &&) noexcept;
    sp_model &operator=(sp_model &&) noexcept;

   ~sp_model();

    std::int64_t
    token_to_index(std::string_view token) const;

    std::string_view
    index_to_token(std::int64_t idx) const;

    std::optional<std::int64_t>
    unk_idx() const;

    std::optional<std::int64_t>
    bos_idx() const;

    std::optional<std::int64_t>
    eos_idx() const;

    std::optional<std::int64_t>
    pad_idx() const;

    std::size_t
    vocabulary_size() const;

    std::string
    serialize() const;

private:
    std::unique_ptr<detail::sp_processor> processor_;
};

}  // namespace fairseq2n
