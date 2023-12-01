// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include "fairseq2n/data/video/detail/ffmpeg.h"

#include "fairseq2n/api.h"
#include "fairseq2n/data/data.h"

#include <ATen/Device.h>
#include <ATen/ScalarType.h>

using namespace fairseq2n::detail;

namespace fairseq2n {

class FAIRSEQ2_API video_decoder {
public:
    explicit
    video_decoder(video_decoder_options opts = {}, bool pin_memory = false);

    data
    operator()(data &&d) const;

private:
    video_decoder_options opts_;
    ffmpeg_decoder decoder_;
        
};

}  // namespace fairseq2n
