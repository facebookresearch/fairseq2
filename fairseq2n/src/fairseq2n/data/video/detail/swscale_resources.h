// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "fairseq2n/api.h"
#include "fairseq2n/data/video/detail/stream.h"

extern "C" {
    #include <libswscale/swscale.h>
}
#include <functional>

namespace fairseq2n::detail {

class FAIRSEQ2_API swscale_resources {
friend class ffmpeg_decoder;

public:
    swscale_resources(int, int, AVPixelFormat);

    ~swscale_resources();

    swscale_resources(const swscale_resources&) = delete;
    
    swscale_resources& operator=(const swscale_resources&) = delete;

private:
    SwsContext *sws_ctx_{nullptr};
};

} // namespace fairseq2n::detail