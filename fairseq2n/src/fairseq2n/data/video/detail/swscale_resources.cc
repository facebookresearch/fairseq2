// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//#include "fairseq2n/data/video/detail/avcodec_resources.h"
//#include "fairseq2n/data/video/detail/utils.h"

#include "fairseq2n/data/video/detail/swscale_resources.h"

#include "fairseq2n/exception.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n::detail {

swscale_resources::swscale_resources(int width, int height, AVPixelFormat fmt)
{
    sws_ctx_ = sws_getContext(width, height, fmt, width, height, AV_PIX_FMT_RGB24,
                                SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (sws_ctx_ == nullptr) {
        throw_<std::runtime_error>("Failed to create the conversion context\n");
    }
}

swscale_resources::~swscale_resources()
{
    if (sws_ctx_ != nullptr)
        sws_freeContext(sws_ctx_);
}

} // namespace fairseq2n::detail