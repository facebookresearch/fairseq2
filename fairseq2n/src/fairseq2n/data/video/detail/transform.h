// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>

#include "fairseq2n/api.h"
#include "fairseq2n/data/video/detail/stream.h"

extern "C" {
    #include <libswscale/swscale.h>
}
#include <functional>

namespace fairseq2n::detail {

class FAIRSEQ2_API transform {
friend class ffmpeg_decoder;

public:
    transform(int width, int height, AVPixelFormat fmt, video_decoder_options opts);

    ~transform();

    void
    transform_to_rgb(AVFrame& sw_frame, const AVFrame &frame, int stream_index, 
    video_decoder_options opts);

    transform(const transform&) = delete;
    
    transform& operator=(const transform&) = delete;

private:
    SwsContext *sws_ctx_{nullptr};
};

} // namespace fairseq2n::detail