// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavformat/avio.h>
    #include <libavutil/avutil.h>
}

using namespace std;

namespace fairseq2n {

struct buffer_data {
    const uint8_t *ptr; // Pointer to the start of the memory_block buffer
    size_t size;        
};

struct media_metadata {
  long num{0}; // Time base numerator
  long den{1}; // Time base denominator
  long duration{-1}; // Duration of the stream, in miscroseconds
  double fps{0}; // Frames per second for video streams
  // media_format format; // TODO
};
}
