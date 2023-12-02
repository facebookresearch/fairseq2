// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include "fairseq2n/data/video/detail/utils.h"
#include "fairseq2n/data/video/detail/stream.h"

#include "fairseq2n/api.h"
#include "fairseq2n/data/data.h"

#include <ATen/Device.h>
#include <ATen/ScalarType.h>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavutil/avutil.h>
    #include <libavutil/imgutils.h>
    #include <libswscale/swscale.h>
    #include <libswresample/swresample.h>
}

namespace fairseq2n::detail {

class FAIRSEQ2_API ffmpeg_decoder {
public:
    explicit
    ffmpeg_decoder(video_decoder_options opts = {});

    ~ffmpeg_decoder();

    data_dict
    open_container(memory_block block);

    data
    open_stream(int stream_index);
    
    static int 
    read_callback(void *opaque, uint8_t *buf, int buf_size);

private:
    video_decoder_options opts_; 
    AVFormatContext* fmt_ctx_ = nullptr;
    AVIOContext* avio_ctx_ = nullptr;
    uint8_t* avio_ctx_buffer_ = nullptr;
    std::unique_ptr<stream> av_stream_; 
};

} // namespace fairseq2n
