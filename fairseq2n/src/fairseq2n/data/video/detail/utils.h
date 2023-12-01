// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

extern "C" {
   #include <libavcodec/avcodec.h>
   #include <libavformat/avformat.h>
   #include <libavformat/avio.h>
   #include <libavutil/avutil.h>
}


namespace fairseq2n::detail {

struct buffer_data {
    const uint8_t *ptr; // Pointer to the start of the memory_block buffer
    size_t size;        
};

struct media_metadata {
  int64_t num_frames{-1}; // Number of frames in the stream
  int numerator{0}; // Time base numerator
  int denominator{0}; // Time base denominator
  int64_t duration_microseconds{-1}; // Duration of the stream
  int height{0}; // Height of a frame in pixels
  int width{0}; // Width of a frame in pixels
  double time_base{0}; // Time base of the stream
  double fps{0}; // Frames per second for video streams
  // media_format format; // TODO
};

/*
static int 
ffmpeg_read_callback(void *opaque, uint8_t *buf, int buf_size) {
    // Read up to buf_size bytes from the resource accessed by the AVIOContext object
    // Used by ffmpeg to read from memory buffer
    fairseq2n::detail::buffer_data *bd = static_cast<fairseq2n::detail::buffer_data *>(opaque);
    buf_size = std::min(buf_size, static_cast<int>(bd->size));
    if (buf_size <= 0)
        return AVERROR_EOF;
    memcpy(buf, bd->ptr, buf_size);
    bd->ptr += buf_size;
    bd->size -= buf_size;
    return buf_size;
};
*/

}
