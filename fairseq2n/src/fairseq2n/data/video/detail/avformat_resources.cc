// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/video/detail/avformat_resources.h"
#include "fairseq2n/data/video/video_decoder.h"

#include "fairseq2n/exception.h"
#include "fairseq2n/detail/exception.h"
#include <functional> 

namespace fairseq2n::detail {

avformat_resources::avformat_resources(size_t data_size, fairseq2n::detail::buffer_data bd)
{
    fmt_ctx_ = avformat_alloc_context();
    if (!(fmt_ctx_)) {
        throw std::runtime_error("Failed to allocate AVFormatContext.");
    }

    avio_ctx_buffer_ = (uint8_t*)av_malloc(data_size + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!avio_ctx_buffer_) {
        throw std::runtime_error("Failed to allocate AVIOContext buffer.");
    }

    auto read_callback_lambda = [](void* opaque, uint8_t* buf, int buf_size) -> int {
        return read_callback(opaque, buf, buf_size);  
    };

    avio_ctx_ = avio_alloc_context(avio_ctx_buffer_, data_size, 0, &bd, read_callback_lambda, nullptr, nullptr);
    if (!avio_ctx_) {
        throw std::runtime_error("Failed to allocate AVIOContext.");
    }
}

avformat_resources::~avformat_resources()
{
    
    if (avio_ctx_) {
        av_freep(&avio_ctx_->buffer);
        av_freep(&avio_ctx_);
    }
    if (fmt_ctx_) {
        avformat_free_context(fmt_ctx_);
    }
    // avio_ctx_bufffer_ is automatically freed by ffmpeg
    
}

AVFormatContext* avformat_resources::get_fmt_ctx() const
{
    return fmt_ctx_;
}

AVIOContext* avformat_resources::get_avio_ctx() const
{
    return avio_ctx_;
}

} // namespace fairseq2n::detail

