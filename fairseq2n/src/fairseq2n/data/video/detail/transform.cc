// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//#include "fairseq2n/data/video/detail/avcodec_resources.h"
//#include "fairseq2n/data/video/detail/utils.h"

#include "fairseq2n/data/video/detail/transform.h"

#include "fairseq2n/exception.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n::detail {

transform::transform(int width, int height, AVPixelFormat fmt, video_decoder_options opts)
{
    int dstWidth = (opts.width() != 0) ? opts.width() : width;
    int dstHeight = (opts.height() != 0) ? opts.height() : height;
    sws_ctx_ = sws_getContext(width, height, fmt, dstWidth, dstHeight, AV_PIX_FMT_RGB24,
                                SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (sws_ctx_ == nullptr) {
        throw_<std::runtime_error>("Failed to create the conversion context\n");
    }
}

void transform::transform_to_rgb(AVFrame& sw_frame, const AVFrame &frame, int stream_index, 
video_decoder_options opts)
{
    // AV_PIX_FMT_RGB24 guarantees 3 color channels
    av_frame_unref(&sw_frame); // Safety check
    sw_frame.format = AV_PIX_FMT_RGB24;
    sw_frame.width = (opts.width() != 0) ? opts.width() : frame.width;
    sw_frame.height = (opts.height() != 0) ? opts.height() : frame.height;
    int ret = av_frame_get_buffer(&sw_frame, 0);
    if (ret < 0) {
        throw_<std::runtime_error>("Failed to allocate buffer for the RGB frame for stream {}\n", 
        stream_index);
    }  
    ret = sws_scale(sws_ctx_, frame.data, frame.linesize, 0, frame.height,
                    sw_frame.data, sw_frame.linesize);
    if (ret < 0) {
        throw_<std::runtime_error>("Failed to convert the frame to RGB for stream {}\n", 
        stream_index);
    }
}

transform::~transform()
{
    if (sws_ctx_ != nullptr)
        sws_freeContext(sws_ctx_);
}

} // namespace fairseq2n::detail