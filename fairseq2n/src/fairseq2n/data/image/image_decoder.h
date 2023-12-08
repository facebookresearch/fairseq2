// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>

#include "fairseq2n/api.h"
#include "fairseq2n/data/data.h"

#include <ATen/Device.h>
#include <ATen/ScalarType.h>

namespace fairseq2n {

class image_decoder_options {
public:
    image_decoder_options
    maybe_device(std::optional<at::Device> value) noexcept
    {
        auto tmp = *this;

        tmp.maybe_device_ = value;

        return tmp;
    }

    std::optional<at::Device>
    maybe_device() const noexcept
    {
        return maybe_device_;
    }

    image_decoder_options
    pin_memory(bool value) noexcept
    {
        auto tmp = *this;

        tmp.pin_memory_ = value;

        return tmp;
    }

    bool
    pin_memory() const noexcept
    {
        return pin_memory_;
    }

private:
    std::optional<at::Device> maybe_device_{};
    bool pin_memory_ = false;
};

class FAIRSEQ2_API image_decoder {
public:
    explicit
    image_decoder(image_decoder_options opts = {});

    data
    operator()(data &&d) const;

private:
    image_decoder_options opts_;

    static bool
    is_little_endian();

    data
    decode_png(const memory_block &block) const;

    data
    decode_jpeg(const memory_block &block) const;
};

}  // namespace fairseq2n
