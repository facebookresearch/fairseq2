// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data.h"

#include <ATen/Device.h>
#include <ATen/ScalarType.h>

namespace fairseq2 {

class audio_decoder_options {
public:
    audio_decoder_options
    dtype(std::optional<at::ScalarType> dtype) && noexcept
    {
        dtype_ = dtype;

        return *this;
    }

    std::optional<at::ScalarType>
    dtype() const noexcept
    {
        return dtype_;
    }

    audio_decoder_options
    device(std::optional<at::Device> value) && noexcept
    {
        device_ = value;

        return *this;
    }

    std::optional<at::Device>
    device() const noexcept
    {
        return device_;
    }

    audio_decoder_options
    pin_memory(bool value) && noexcept
    {
        pin_memory_ = value;

        return *this;
    }

    bool
    pin_memory() const noexcept
    {
        return pin_memory_;
    }

private:
    std::optional<at::ScalarType> dtype_{};
    std::optional<at::Device> device_{};
    bool pin_memory_ = false;
};

class FAIRSEQ2_API audio_decoder {
public:
    explicit
    audio_decoder(audio_decoder_options opts = {});

    data
    operator()(data &&d) const;

private:
    audio_decoder_options opts_;
};

}  // namespace fairseq2
