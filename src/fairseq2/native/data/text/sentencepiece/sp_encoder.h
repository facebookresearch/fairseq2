// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <ATen/Device.h>
#include <ATen/ScalarType.h>
#include <ATen/Tensor.h>

#include "fairseq2/native/api.h"
#include "fairseq2/native/float.h"
#include "fairseq2/native/span.h"
#include "fairseq2/native/data/data.h"

namespace fairseq2 {

class sp_encoder_options {
public:
    sp_encoder_options
    prefix_token(std::string value) &&
    {
        prefix_tokens_.push_back(std::move(value));

        return std::move(*this);
    }

    std::vector<std::string> &
    prefix_tokens() noexcept
    {
        return prefix_tokens_;
    }

    const std::vector<std::string> &
    prefix_tokens() const noexcept
    {
        return prefix_tokens_;
    }

    sp_encoder_options
    suffix_token(std::string value) &&
    {
        suffix_tokens_.push_back(std::move(value));

        return std::move(*this);
    }

    std::vector<std::string> &
    suffix_tokens() noexcept
    {
        return suffix_tokens_;
    }

    const std::vector<std::string> &
    suffix_tokens() const noexcept
    {
        return suffix_tokens_;
    }

    sp_encoder_options
    reverse(bool value) && noexcept
    {
        reverse_ = value;

        return std::move(*this);
    }

    bool
    reverse() const noexcept
    {
        return reverse_;
    }

    sp_encoder_options
    enable_sampling(bool value) && noexcept
    {
        enable_sampling_ = value;

        return std::move(*this);
    }

    bool
    enable_sampling() const noexcept
    {
        return enable_sampling_;
    }

    sp_encoder_options
    nbest_size(std::int32_t value) && noexcept
    {
        nbest_size_ = value;

        return std::move(*this);
    }

    std::int32_t
    nbest_size() const noexcept
    {
        return nbest_size_;
    }

    sp_encoder_options
    alpha(float32 value) && noexcept
    {
        alpha_ = value;

        return std::move(*this);
    }

    float32
    alpha() const noexcept
    {
        return alpha_;
    }

    sp_encoder_options
    maybe_device(std::optional<at::Device> value) && noexcept
    {
        maybe_device_ = value;

        return std::move(*this);
    }

    std::optional<at::Device>
    maybe_device() const noexcept
    {
        return maybe_device_;
    }

    sp_encoder_options
    pin_memory(bool value) && noexcept
    {
        pin_memory_ = value;

        return std::move(*this);
    }

    bool
    pin_memory() const noexcept
    {
        return pin_memory_;
    }

private:
    std::vector<std::string> prefix_tokens_{};
    std::vector<std::string> suffix_tokens_{};
    bool reverse_{};
    bool enable_sampling_{};
    std::int32_t nbest_size_ = -1;
    float32 alpha_ = 0.1F;
    std::optional<at::Device> maybe_device_{};
    bool pin_memory_ = false;
};

namespace detail {

class encoder_op;

}

class immutable_string;

class sp_model;

class FAIRSEQ2_API sp_encoder final {
    friend class detail::encoder_op;

public:
    explicit
    sp_encoder(std::shared_ptr<const sp_model> model, sp_encoder_options opts = {});

    data
    operator()(data &&d) const;

private:
    at::Tensor
    encode(immutable_string &&sentence) const;

private:
    std::shared_ptr<const sp_model> model_;
    sp_encoder_options opts_;
    std::vector<std::int64_t> prefix_token_indices_{};
    std::vector<std::int64_t> suffix_token_indices_{};
};

}  // namespace fairseq2
