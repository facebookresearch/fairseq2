// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>

#include "fairseq2n/api.h"
#include "fairseq2n/data/data.h"

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavformat/avio.h>
    #include <libavutil/avutil.h>
    #include <libswscale/swscale.h>
}

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

    data
    operator()(data &&d);

    int
    open_container(memory_block block);

    int 
    open_streams();

    int
    decode_frame(int stream_index);

    static int
    read_callback(void *opaque, uint8_t *buf, int buf_size);

    // TODO
    static int
    seek_callback(void *opaque, int64_t offset, int whence);

    void
    clean();

private:
    video_decoder_options opts_;
    AVFormatContext* fmt_ctx_{nullptr};
    AVIOContext *avio_ctx_{nullptr};
    AVCodecContext *codec_ctx_{nullptr};
    AVCodec *codec_{nullptr};
    AVCodecParameters *codec_par_{nullptr};
    uint8_t *avio_ctx_buffer_{nullptr};
    struct SwsContext *pImgConvertCtx;
    int width_; // width of the video frame
    int height_; // height of the video frame 
    double fps_; // frames per second
    double video_timebase_; // time base of the video stream
    double audio_timebase_; // time base of the audio stream
    
};

}  // namespace fairseq2n
