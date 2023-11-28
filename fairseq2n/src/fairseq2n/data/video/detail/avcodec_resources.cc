// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/video/detail/avcodec_resources.h"
#include "fairseq2n/data/video/detail/utils.h"

#include "fairseq2n/exception.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n::detail {

avcodec_resources::avcodec_resources(AVCodec* codec)
{
    codec_ctx_ = avcodec_alloc_context3(codec);
        if (!codec_ctx_) {
            throw std::runtime_error("Failed to allocate the decoder context for stream.");
        }
}

avcodec_resources::~avcodec_resources()
{
    if (codec_ctx_)
        avcodec_free_context(&codec_ctx_);
}

AVCodecContext* avcodec_resources::get_codec_ctx() const
{
    return codec_ctx_;
}

} // namespace fairseq2n::detail
