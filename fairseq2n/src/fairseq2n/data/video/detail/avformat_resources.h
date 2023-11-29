// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "fairseq2n/data/video/detail/utils.h"

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavformat/avio.h>
}
#include <functional>

namespace fairseq2n::detail {

class avformat_resources {
friend class ffmpeg_decoder;

public:
    avformat_resources(size_t data_size, fairseq2n::detail::buffer_data bd);
    ~avformat_resources();
    AVFormatContext* get_fmt_ctx() const;
    AVIOContext* get_avio_ctx() const;
    avformat_resources(const avformat_resources&) = delete;
    avformat_resources& operator=(const avformat_resources&) = delete;

private: 
    AVFormatContext* fmt_ctx_ = nullptr;
    AVIOContext* avio_ctx_ = nullptr;
    uint8_t* avio_ctx_buffer_ = nullptr;
    //friend class fairseq2n::ffmpeg_decoder;
};

} // namespace fairseq2n::detail
