// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/video/detail/ffmpeg.h"

#include <cstdint>
#include <exception>
#include <stdexcept>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include "fairseq2n/exception.h"
#include "fairseq2n/float.h"
#include "fairseq2n/fmt.h"
#include "fairseq2n/memory.h"
#include "fairseq2n/data/detail/tensor_helpers.h"
#include "fairseq2n/detail/exception.h"

using namespace std;

namespace fairseq2n::detail {

ffmpeg_decoder::ffmpeg_decoder(video_decoder_options opts)
    : opts_{opts}
{}

data_dict
ffmpeg_decoder::open_container(const memory_block &block)
{
    // Opens the media container and iterates over streams.

    auto data_ptr = reinterpret_cast<const uint8_t*>(block.data());
    av_register_all();
    size_t data_size = block.size();
    fairseq2n::detail::buffer_data bd = {data_ptr, data_size};   
    int ret = 0;
    
    fmt_ctx_ = avformat_alloc_context();
    if (fmt_ctx_ == nullptr) {
        throw_<runtime_error>("Failed to allocate AVFormatContext.");
    }
    // Allocate buffer for input/output operations via AVIOContext
    avio_ctx_buffer_ = static_cast<uint8_t*>(av_malloc(data_size + AV_INPUT_BUFFER_PADDING_SIZE));
    if (avio_ctx_buffer_ == nullptr) {
        throw_<runtime_error>("Failed to allocate AVIOContext buffer.");
    }
    // Create an AVIOContext for using custom IO
    avio_ctx_ = avio_alloc_context(
        avio_ctx_buffer_, 
        data_size, 
        0, // Write flag
        &bd, // Pointer to user data
        &read_callback, // Read function
        nullptr, // Write function, not used
        nullptr // Seek function, not used
        );
    if (avio_ctx_ == nullptr) {
        throw_<runtime_error>("Failed to allocate AVIOContext.");
    }
    
    fmt_ctx_->pb = avio_ctx_; 
    fmt_ctx_->flags |= AVFMT_FLAG_CUSTOM_IO; 
    fmt_ctx_->flags |= AVFMT_FLAG_NONBLOCK;
    
    // Determine the input format
    AVProbeData probe_data = {0};
    probe_data.buf = avio_ctx_buffer_;
    probe_data.buf_size = data_size;
    probe_data.filename = "";  // Set to an empty string since we don't have a filename
    fmt_ctx_->iformat = av_probe_input_format(&probe_data, 1);
    
    // Open media file and read the header
    ret = avformat_open_input(&fmt_ctx_, nullptr, fmt_ctx_->iformat, nullptr);
    if (ret < 0) {
        throw_with_nested<invalid_argument>("Failed to open input.");
    }

    // Read data from the media file
    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (ret < 0) {
        throw_<runtime_error>("Failed to find stream information.");
    }

    // Iterate over all streams
    flat_hash_map<std::string, data> all_streams;
    for (int i = 0; i < static_cast<int>(fmt_ctx_->nb_streams); i++) {
        all_streams[std::to_string(i)] = open_stream(i);
    }

    return all_streams;
}   

data_dict
ffmpeg_decoder::open_stream(int stream_index) 
{
    // Opens a stream and decodes the video frames. Skips audio streams for now.

    av_stream_ = std::make_unique<stream>(stream_index, *fmt_ctx_);
    int processed_frames = 0;
    if (av_stream_->type_ == AVMEDIA_TYPE_VIDEO) {
        av_stream_->alloc_resources();
    
        // Fill codec context with codec parameters
        int ret = avcodec_parameters_to_context(av_stream_->codec_ctx_, av_stream_->codec_params_);
        if (ret < 0) {
            throw_<runtime_error>("Failed to copy decoder parameters to input decoder context for stream {}\n", 
            stream_index);
        }

        // Open the codec
        ret = avcodec_open2(av_stream_->codec_ctx_, av_stream_->codec_, nullptr);
        if (ret < 0) {
            throw_<runtime_error>("Failed to open decoder for stream {}\n", stream_index);  
        }
    
        // Create tensor storage for the stream
        av_stream_->init_tensor_storage(opts_);
        // Iterate over all frames in the stream and decode them
        while (av_read_frame(fmt_ctx_, av_stream_->pkt_) >= 0) {       
            if (av_stream_->pkt_->stream_index == stream_index) {  
                // Send raw data packet (compressed frame) to the decoder through the codec context
                ret = avcodec_send_packet(av_stream_->codec_ctx_, av_stream_->pkt_);
                if (ret < 0) {
                    throw_<runtime_error>("Error sending packet to decoder for stream {}\n", 
                    stream_index);
                }
                // Receive raw data frame (uncompressed frame) from the decoder through the codec context
                while (ret >= 0) {
                    ret = avcodec_receive_frame(av_stream_->codec_ctx_, av_stream_->frame_);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;  
                        // EAGAIN is not an error, it means we need more input
                        // AVERROR_EOF means decoding finished
                    } else if (ret < 0) {
                        throw_<runtime_error>("Error receiving frame from decoder for stream {}\n", 
                        stream_index);
                    }                                                                                          
                    // Tranform frame to RGB to guarantee 3 color channels
                    sws_ = std::make_unique<transform>(av_stream_->frame_->width, av_stream_->frame_->height, 
                                                static_cast<AVPixelFormat>(av_stream_->frame_->format), opts_);
                    sws_->transform_to_rgb(*av_stream_->sw_frame_, *av_stream_->frame_, stream_index, opts_);
                    // Store PTS in microseconds
                    if (!opts_.get_frames_only()) {
                        av_stream_->tensor_storage_.frame_pts[processed_frames] = av_stream_->frame_->pts * av_stream_->metadata_.time_base * 1000000;
                    }
                    // Store raw frame data for one frame
                    if (!opts_.get_pts_only()) {
                        at::Tensor one_frame = av_stream_->tensor_storage_.all_video_frames[processed_frames];                        
                        writable_memory_span frame_bits = get_raw_mutable_storage(one_frame);
                        auto frame_data = reinterpret_cast<uint8_t*>(frame_bits.data());
                        // Calculate the total size of the frame in bytes
                        int frame_size = av_image_get_buffer_size(AV_PIX_FMT_RGB24, av_stream_->sw_frame_->width, 
                                                                                av_stream_->sw_frame_->height, 1);
                        // Copy the entire frame at once                
                        memcpy(frame_data, av_stream_->sw_frame_->data[0], frame_size);
                    }
                    processed_frames++;
                                                               
                    av_frame_unref(av_stream_->frame_); // Unref old data so the frame can be reused
                    av_frame_unref(av_stream_->sw_frame_);
                }
            }
            av_packet_unref(av_stream_->pkt_);   
        }
        av_stream_->init_data_storage(opts_);

        return av_stream_->stream_data_;
    } else {
        // Skip streams if not video for now
        // Return an empty data object if the stream is not video
        return data_dict{};
    }
}

int 
ffmpeg_decoder::read_callback(void *opaque, uint8_t *buf, int buf_size) 
{
    // Read up to buf_size bytes from the resource accessed by the AVIOContext object
    // Used by ffmpeg to read from memory buffer
    auto *bd = static_cast<fairseq2n::detail::buffer_data *>(opaque);
    buf_size = std::min(static_cast<size_t>(buf_size), bd->size);
    if (buf_size <= 0)
        return AVERROR_EOF;
    memcpy(buf, bd->ptr, buf_size);
    bd->ptr += buf_size;
    bd->size -= static_cast<size_t>(buf_size);
    return buf_size;
}

ffmpeg_decoder::~ffmpeg_decoder()
{   
    if (avio_ctx_ != nullptr) {
        av_freep(&avio_ctx_->buffer);
        av_freep(&avio_ctx_);
    }
    if (fmt_ctx_ != nullptr) {
        avformat_free_context(fmt_ctx_);
    }
}

} // namespace fairseq2n
