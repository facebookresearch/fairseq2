// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <decord/video_interface.h>
#include <optional>

#include "fairseq2n/api.h"
#include "fairseq2n/data/data.h"

#include <ATen/Device.h>
#include <ATen/ScalarType.h>

namespace fairseq2n {

class video_decoder_options {
public:
    video_decoder_options
    maybe_dtype(std::optional<at::ScalarType> value) noexcept
    {
        auto tmp = *this;

        tmp.maybe_dtype_ = value;

        return tmp;
    }

    std::optional<at::ScalarType>
    maybe_dtype() const noexcept
    {
        return maybe_dtype_;
    }

    video_decoder_options
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

    video_decoder_options
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
    std::optional<at::ScalarType> maybe_dtype_{};
    std::optional<at::Device> maybe_device_{};
    bool pin_memory_ = false;
};

class FAIRSEQ2_API video_decoder {
public:
    explicit
    video_decoder(video_decoder_options opts = {}, bool pin_memory = false);

    struct decoder_metadata {
    unsigned long format;
    };

    struct decoder_header {
    unsigned long format;
    };

    class byte_storage {
    public:
    virtual ~byte_storage() = default;
    virtual size_t length() const = 0;
    };

    struct decoder_output {
    decoder_header header;
    std::unique_ptr<byte_storage> payload;
    };

    data
    operator()(data &&d) const;

    static int
    read_callback(void *opaque, uint8_t *buf, int buf_size);

    static int
    seek_callback(void *opaque, int64_t offset, int whence);

    int 
    decode_video(decoder_output* out);

    int
    decode_frame();

private:
    video_decoder_options opts_;
    std::list<decoder_output> queue_;
    bool eof_{false};
};

}  // namespace fairseq2n
