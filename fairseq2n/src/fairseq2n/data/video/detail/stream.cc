// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/video/detail/stream.h"
#include "fairseq2n/data/video/detail/utils.h"

#include "fairseq2n/exception.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n::detail {

stream::stream(int stream_index, AVFormatContext* fmt_ctx) {
    // Initialize the AVStream, AVCodecParameters, and get metadata 
    stream_index_ = stream_index;
    av_stream_ = fmt_ctx->streams[stream_index];
    metadata_.numerator = av_stream_->time_base.num;
    metadata_.denominator = av_stream_->time_base.den;
    metadata_.duration_microseconds = fmt_ctx->duration;
    metadata_.fps = av_q2d(av_stream_->avg_frame_rate);
    metadata_.num_frames = av_stream_->nb_frames;
    metadata_.time_base = av_q2d(av_stream_->time_base);
    codec_params_ = av_stream_->codecpar;
    metadata_.height = codec_params_->height;
    metadata_.width = codec_params_->width;
    type_ = codec_params_->codec_type;
}

void
stream::alloc_resources() {
    // Allocate memory to hold the context for decoding process
    codec_ctx_ = avcodec_alloc_context3(codec_);
    if (!codec_ctx_) {
       throw std::runtime_error("Failed to allocate the decoder context for stream.");
    }
    // Allocate memory to hold the packet
    pkt_ = av_packet_alloc();
    if (!pkt_) {
        fprintf(stderr, "Failed to allocate the packet\n");
        throw std::runtime_error("Failed to allocate the packet.");
    }
    // Allocate memory to hold the frames
    frame_ = av_frame_alloc();
    if (!frame_) {
        fprintf(stderr, "Failed to allocate the frame\n");
        throw std::runtime_error("Failed to allocate the frame.");
    }
    sw_frame_ = av_frame_alloc();
    if (!sw_frame_) {
        fprintf(stderr, "Failed to allocate the software frame\n");
        throw std::runtime_error("Failed to allocate the software frame.");
    }
}

void
stream::find_codec() {
    // Find the decoder for the stream
    codec_ = avcodec_find_decoder(codec_params_->codec_id);
    if (!codec_) {
        fprintf(stderr, "Failed to find decoder for stream #%u\n", stream_index_);
        throw std::runtime_error("Failed to find decoder for stream.");
    }
}

void
stream::init_tensor_storage(bool pin_memory) {
    // Initialize tensors for storing raw frames and metadata
    storage_.all_video_frames = at::empty({metadata_.num_frames, metadata_.height, metadata_.width, 3},
    at::dtype(at::kByte).device(at::kCPU).pinned_memory(pin_memory));
    storage_.frame_pts = at::empty({metadata_.num_frames},
    at::dtype(at::kLong).device(at::kCPU).pinned_memory(pin_memory));
    storage_.timebase = at::tensor({metadata_.numerator, metadata_.denominator},
    at::dtype(at::kInt).device(at::kCPU).pinned_memory(pin_memory));
    storage_.fps = at::tensor({metadata_.fps},
    at::dtype(at::kFloat).device(at::kCPU).pinned_memory(pin_memory));
    storage_.duration = at::tensor({metadata_.duration_microseconds},
    at::dtype(at::kLong).device(at::kCPU).pinned_memory(pin_memory));
}

stream::~stream() {
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
    }
    if (frame_) {
        av_frame_free(&frame_);
    }
    if (sw_frame_) {
        av_frame_free(&sw_frame_);
    }
    if (pkt_) {
        av_packet_free(&pkt_);
    }
}

} // namespace fairseq2n::detail
