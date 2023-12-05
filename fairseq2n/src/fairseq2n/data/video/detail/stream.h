// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "fairseq2n/api.h"
#include "fairseq2n/data/video/detail/utils.h"

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <optional>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
}
#include <functional>
#include "fairseq2n/memory.h"
#include "fairseq2n/data/detail/tensor_helpers.h"

namespace fairseq2n::detail {

class FAIRSEQ2_API stream {
friend class ffmpeg_decoder;
public:
    stream(int stream_index, const AVFormatContext& fmt_ctx);

    void 
    alloc_resources();

    ~stream();

    void 
    init_tensor_storage(video_decoder_options opts);

    AVCodecContext* get_codec_ctx() const;

    stream(const stream&) = delete;

    stream& operator=(const stream&) = delete;

private:
    AVCodecContext* codec_ctx_{nullptr};
    AVFrame* frame_{nullptr};
    AVFrame *sw_frame_{nullptr};
    AVStream *av_stream_{nullptr};
    AVPacket *pkt_{nullptr};
    media_metadata metadata_;
    AVCodecParameters* codec_params_{nullptr};
    AVMediaType type_{AVMEDIA_TYPE_UNKNOWN};
    AVCodec* codec_;
    int stream_index_;
    tensor_storage storage_;
};

} // namespace fairseq2n::detail
