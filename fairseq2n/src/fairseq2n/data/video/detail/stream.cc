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

stream::stream(int stream_index, const AVFormatContext& fmt_ctx) {
    // Initialize the AVStream, AVCodecParameters, AVCodec, and get metadata 
    stream_index_ = stream_index;
    av_stream_ = fmt_ctx.streams[stream_index];
    metadata_.numerator = av_stream_->time_base.num;
    metadata_.denominator = av_stream_->time_base.den;
    metadata_.duration_microseconds = fmt_ctx.duration;
    metadata_.fps = av_q2d(av_stream_->avg_frame_rate);
    metadata_.num_frames = av_stream_->nb_frames;
    metadata_.time_base = av_q2d(av_stream_->time_base);
    codec_params_ = av_stream_->codecpar;
    metadata_.height = codec_params_->height;
    metadata_.width = codec_params_->width;
    type_ = codec_params_->codec_type;
    codec_ = avcodec_find_decoder(codec_params_->codec_id);
    if (codec_ == nullptr) {
        throw_<std::runtime_error>("Failed to find decoder for stream {}\n", 
        stream_index_);
    }
}

void
stream::alloc_resources() {
    // Allocate memory to hold the context for decoding process
    codec_ctx_ = avcodec_alloc_context3(codec_);
    if (codec_ctx_ == nullptr) {
       throw_<std::runtime_error>("Failed to allocate the decoder context for stream {}\n", 
       stream_index_);
    }
    // Allocate memory to hold the packet
    pkt_ = av_packet_alloc();
    if (pkt_ == nullptr) {
        throw_<std::runtime_error>("Failed to allocate the packet for stream {}\n", 
        stream_index_);
    }
    // Allocate memory to hold the frames
    frame_ = av_frame_alloc();
    if (frame_ == nullptr) {
        throw_<std::runtime_error>("Failed to allocate the frame for stream {}\n", 
        stream_index_);
    }
    sw_frame_ = av_frame_alloc();
    if (sw_frame_ == nullptr) {
        throw_<std::runtime_error>("Failed to allocate the software frame for stream {}\n", 
        stream_index_);
    }
}

void
stream::init_tensor_storage(video_decoder_options opts) {
    // Initialize tensors for storing raw frames and metadata

    if (!opts.get_pts_only()) {
        tensor_storage_.all_video_frames = at::empty({metadata_.num_frames, metadata_.height, metadata_.width, 3},
        at::dtype(opts.maybe_dtype().value_or(at::kByte)).device(at::kCPU).pinned_memory(opts.pin_memory()));
    }

    if (!opts.get_frames_only()) {
        tensor_storage_.frame_pts = at::empty({metadata_.num_frames},
        at::dtype(at::kLong).device(at::kCPU).pinned_memory(opts.pin_memory()));
    }

    if (!opts.get_pts_only() && !opts.get_frames_only()) {
        tensor_storage_.timebase = at::tensor({metadata_.numerator, metadata_.denominator},
        at::dtype(at::kInt).device(at::kCPU).pinned_memory(opts.pin_memory()));
        tensor_storage_.fps = at::tensor({metadata_.fps},
        at::dtype(at::kFloat).device(at::kCPU).pinned_memory(opts.pin_memory()));
        tensor_storage_.duration = at::tensor({metadata_.duration_microseconds},
        at::dtype(at::kLong).device(at::kCPU).pinned_memory(opts.pin_memory()));
    }
}

void
stream::init_data_storage(video_decoder_options opts) {

    if (!opts.get_pts_only()) {
        stream_data_["all_video_frames"] = data(tensor_storage_.all_video_frames);
    }

    if (!opts.get_frames_only()) {
        stream_data_["frame_pts"] = data(tensor_storage_.frame_pts); 
    }

    if (!opts.get_pts_only() && !opts.get_frames_only()) {
        stream_data_["timebase"] = data(tensor_storage_.timebase);
        stream_data_["fps"] = data(tensor_storage_.fps);
        stream_data_["duration"] = data(tensor_storage_.duration);   
    }
}

stream::~stream() {
    if (codec_ctx_ != nullptr) {
        avcodec_free_context(&codec_ctx_);
    }
    if (frame_ != nullptr) {
        av_frame_free(&frame_);
    }
    if (sw_frame_ != nullptr) {
        av_frame_free(&sw_frame_);
    }
    if (pkt_ != nullptr) {
        av_packet_free(&pkt_);
    }
}

} // namespace fairseq2n::detail
