// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

extern "C" {
    #include <libavcodec/avcodec.h>
}
#include <functional>

namespace fairseq2n::detail {

class avcodec_resources {
friend class ffmpeg_decoder;

public:
    avcodec_resources(AVCodec *codec);
    ~avcodec_resources();
    AVCodecContext* get_codec_ctx() const;
    avcodec_resources(const avcodec_resources&) = delete;
    avcodec_resources& operator=(const avcodec_resources&) = delete;

private:
    AVCodecContext* codec_ctx_{nullptr};
    //friend class ffmpeg_decoder;
};

} // namespace fairseq2n::detail
