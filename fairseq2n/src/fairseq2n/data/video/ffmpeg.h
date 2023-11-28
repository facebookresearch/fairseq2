// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include "fairseq2n/data/video/detail/avcodec_resources.h"
#include "fairseq2n/data/video/detail/utils.h"
#include "fairseq2n/data/video/detail/avformat_resources.h"

#include "fairseq2n/api.h"
#include "fairseq2n/data/data.h"

#include <ATen/Device.h>
#include <ATen/ScalarType.h>

extern "C" {
    #include <libavcodec/avcodec.h>
    //#include <libavformat/avformat.h>
    #include <libavformat/avio.h>
    #include <libavutil/avutil.h>
    #include <libavutil/imgutils.h>
    #include <libswscale/swscale.h>
    #include <libswresample/swresample.h>
}

using namespace fairseq2n::detail;

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


class FAIRSEQ2_API ffmpeg_decoder {
public:
    explicit
    ffmpeg_decoder(video_decoder_options opts = {}, bool pin_memory = false);

    at::List<at::List<at::Tensor>>
    open_container(memory_block block);

    at::List<at::Tensor>
    open_stream(int stream_index);

private:
    video_decoder_options opts_; 
    std::unique_ptr<avformat_resources> fmt_resources_;     
    std::unique_ptr<avcodec_resources> codec_resources_; 
};


} // namespace fairseq2n