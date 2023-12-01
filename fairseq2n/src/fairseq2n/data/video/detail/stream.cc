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
    /*
    all_frames_ = at::empty({metadata_.num_frames, metadata_.height, metadata_.width, 3}, 
    at::dtype(at::kByte).device(at::kCPU).pinned_memory(false));
    pts_ = at::empty({metadata_.num_frames}, 
    at::dtype(at::kLong).device(at::kCPU).pinned_memory(false));
    */
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
    codec_ = avcodec_find_decoder(codec_params_->codec_id);
    if (!codec_) {
        fprintf(stderr, "Failed to find decoder for stream #%u\n", stream_index_);
        throw std::runtime_error("Failed to find decoder for stream.");
    }
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

/*
int
stream::process_packet(int stream_index, AVFormatContext* fmt_ctx_) {
    std::cout << "check 4" << std::endl;
    
    int processed_frames = 0;

    int ret = avcodec_parameters_to_context(codec_ctx_, codec_params_);
    if (ret < 0) {
        fprintf(stderr, "Failed to copy decoder parameters to input decoder context "
                "for stream #%u\n", stream_index);
        throw std::runtime_error("Failed to copy decoder parameters to input decoder context.");
    }


    ret = avcodec_open2(codec_ctx_, codec_, nullptr);
        if (ret < 0) {
            fprintf(stderr, "Failed to open decoder for stream #%u\n", stream_index);
            throw std::runtime_error("Failed to open decoder for stream.");
        }
    

    //int processed_frames = 0;
    std::cout << "check 5" << std::endl;
    while (av_read_frame(fmt_ctx_, pkt_) >= 0) { 
        if (pkt_->stream_index == stream_index) {  
                    // Send raw data packet (compressed frame) to the decoder through the codec context
                    int ret = avcodec_send_packet(codec_ctx_, pkt_);
                    if (ret < 0) {
                        fprintf(stderr, "Error sending packet to decoder: %s\n");
                        throw std::runtime_error("Error sending packet to decoder.");
                    }
                    // Receive raw data frame (uncompressed frame) from the decoder through the codec context
                    while (ret >= 0) {
                        ret = avcodec_receive_frame(codec_ctx_, frame_);
                        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                            break;  // EAGAIN is not an error, it means we need more input
                            // AVERROR_EOF means decoding finished
                        } else if (ret < 0) {
                            fprintf(stderr, "Error receiving frame from decoder: %s\n");
                            throw std::runtime_error("Error receiving frame from decoder.");
                        }                                                                    
                        // Save the frame in a tensor
                        if (codec_ctx_->codec_type == AVMEDIA_TYPE_VIDEO) {                            
                            // AV_PIX_FMT_RGB24 guarantees 3 color channels
                            SwsContext *sws_ctx = sws_getContext(frame_->width, frame_->height, static_cast<AVPixelFormat>(frame_->format),
                                                                frame_->width, frame_->height, AV_PIX_FMT_RGB24,
                                                                SWS_BILINEAR, nullptr, nullptr, nullptr);
                            if (!sws_ctx) {
                                fprintf(stderr, "Failed to create the conversion context\n");
                                throw std::runtime_error("Failed to create the conversion context.");
                            }
                            sw_frame_->format = AV_PIX_FMT_RGB24;
                            sw_frame_->width = frame_->width;
                            sw_frame_->height = frame_->height;
                            ret = av_frame_get_buffer(sw_frame_, 0);
                            if (ret < 0) {
                                fprintf(stderr, "Failed to allocate buffer for the RGB frame\n");
                                av_frame_free(&sw_frame_);
                                throw std::runtime_error("Failed to allocate buffer for the RGB frame.");
                            }  
                            ret = sws_scale(sws_ctx, frame_->data, frame_->linesize, 0, frame_->height,
                                            sw_frame_->data, sw_frame_->linesize);
                            if (ret < 0) {
                                fprintf(stderr, "Failed to convert the frame to RGB\n");
                                throw std::runtime_error("Failed to convert the frame to RGB.");
                            }
                            int channels = 3;
                            // Store PTS in microseconds
                            pts_[processed_frames] = (int64_t)(frame_->pts * metadata_.time_base * 1000000);
                            at::Tensor one_frame = all_frames_[processed_frames];                           
                            writable_memory_span frame_bits = get_raw_mutable_storage(one_frame);
                            auto frame_data = reinterpret_cast<uint8_t*>(frame_bits.data());
                            // Calculate the total size of the frame in bytes
                            int frame_size = av_image_get_buffer_size(AV_PIX_FMT_RGB24, sw_frame_->width, sw_frame_->height, 1);
                    
                            // Copy the entire frame at once                
                            memcpy(frame_data, sw_frame_->data[0], frame_size);
                    
                            sws_freeContext(sws_ctx);
                            processed_frames++; 
                        } 
                                                
                        av_frame_unref(frame_); // Unref old data so the frame can be reused
                        av_frame_unref(sw_frame_);
                    }
                }
                av_packet_unref(pkt_); // Unref old data so the packet can be reused

    }
            std::cout << "check 7" << std::endl;
}
*/

} // namespace fairseq2n::detail