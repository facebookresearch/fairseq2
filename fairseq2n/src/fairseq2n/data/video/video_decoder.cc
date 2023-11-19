// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/video/detail/utils.h"

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <iostream>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include "fairseq2n/data/video/video_decoder.h"
#include "fairseq2n/exception.h"
#include "fairseq2n/float.h"
#include "fairseq2n/fmt.h"
#include "fairseq2n/memory.h"
#include "fairseq2n/data/detail/tensor_helpers.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n {

video_decoder::video_decoder(video_decoder_options opts, bool pin_memory)
    : opts_{opts}
{
    at::ScalarType dtype = opts_.maybe_dtype().value_or(at::kFloat);
    if (dtype != at::kFloat && dtype != at::kInt && dtype != at::kShort)
        throw not_supported_error(
            "`video_decoder` supports only `torch.float32`, `torch.int32`, and `torch.int16` data types.");
    
}

data
video_decoder::operator()(data &&d) const
{
    if (!d.is_memory_block())
        throw std::invalid_argument(fmt::format(
            "The input data must be of type `memory_block`, but is of type `{}` instead.", d.type()));

    const memory_block &block = d.as_memory_block();
    if (block.empty())
        throw std::invalid_argument("The input memory block has zero length and cannot be decoded.");

    std::vector<std::vector<at::Tensor>> all_streams = open_container(block);
    
    // Temporarily check content of tensor
    for (const auto& one_stream : all_streams) {
        // Iterate over each tensor in the stream
        for (const auto& tensor : one_stream) {
            // Print the tensor
            std::cout << "Tensor: " << tensor << "\n";

        }
    }
    
    data_dict output;
    //output.emplace("video", std::move(decoded_video));
    return output;

} 

std::vector<std::vector<at::Tensor>>
video_decoder::open_container(memory_block block) const
{
    // Opens the media container and reads the metadata.
    
    auto data_ptr = reinterpret_cast<const uint8_t*>(block.data());
    //av_register_all();
    size_t data_size = block.size();
    fairseq2n::detail::buffer_data bd = {0};   
    bd.ptr = data_ptr;
    bd.size = data_size;
    AVFormatContext* fmt_ctx = nullptr;
    AVIOContext *avio_ctx = nullptr;
    uint8_t* avio_ctx_buffer = nullptr;
    int ret = 0;
    at::ScalarType dtype = opts_.maybe_dtype().value_or(at::kByte);

    // Allocate memory to hold info about the container format
    fmt_ctx = avformat_alloc_context();
    if (!(fmt_ctx)) {
        avformat_free_context(fmt_ctx);
        throw std::runtime_error("Failed to allocate AVFormatContext.");
    }
    // Allocate memory for the buffer holding the data to be read
    avio_ctx_buffer = (uint8_t*)av_malloc(data_size + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!avio_ctx_buffer) {
        avformat_close_input(&fmt_ctx);
        throw std::runtime_error("Failed to allocate AVIOContext buffer.");
    }
    // Create the AVIOContext for accessing the data in the buffer
    avio_ctx = avio_alloc_context(
        avio_ctx_buffer, 
        data_size, 
        0, 
        &bd, 
        &video_decoder::read_callback, 
        nullptr, 
        nullptr);
    if (!avio_ctx) {
        avformat_close_input(&fmt_ctx);
        av_freep(&avio_ctx_buffer);
        throw std::runtime_error("Failed to allocate AVIOContext.");
    }	
    fmt_ctx->pb = avio_ctx; 
    fmt_ctx->flags |= AVFMT_FLAG_CUSTOM_IO; 
    fmt_ctx->flags |= AVFMT_FLAG_NONBLOCK;
    // TODO: Determine the input format, currently causes seg fauit
    /*
    AVProbeData probe_data = {0};
    probe_data.buf = avio_ctx_buffer;
    probe_data.buf_size = data_size;
    probe_data.filename = "";  // Set to an empty string since we don't have a filename
    
    // Determine the input format
    fmt_ctx->iformat = av_probe_input_format(&probe_data, 1);
    */
    // Open media file and read the header
    ret = avformat_open_input(&fmt_ctx, nullptr, nullptr, nullptr);
    if (ret < 0) {
        fprintf(stderr, "Could not open input\n");
        avformat_close_input(&fmt_ctx);
        av_freep(&avio_ctx->buffer);
        avio_context_free(&avio_ctx);
        throw std::runtime_error("Failed to open input.");
    }
    // Read data from the media file
    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        fprintf(stderr, "Could not find stream information\n");
        avformat_close_input(&fmt_ctx);
        av_freep(&avio_ctx->buffer);
        avio_context_free(&avio_ctx);
        throw std::runtime_error("Failed to find stream information.");
    }

    //av_dump_format(fmt_ctx, 0, nullptr, 0);
    std::vector<std::vector<at::Tensor>> all_streams;
    AVPacket *pkt = av_packet_alloc();
    if (!pkt) {
        fprintf(stderr, "Failed to allocate the packet\n");
        throw std::runtime_error("Failed to allocate the packet.");
    }
    // Allocate memory for stream frames
    AVFrame *frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Failed to allocate the frame\n");
        throw std::runtime_error("Failed to allocate the frame.");
    }
    // Allocate memory for RGB frames
    AVFrame *rgb_frame = av_frame_alloc();
    if (!rgb_frame) {
        fprintf(stderr, "Failed to allocate the RGB frame\n");
        throw std::runtime_error("Failed to allocate the RGB frame.");
    }
   
    // Iterate over all streams
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
         // Allocate memory for stream packets
        AVStream *stream = fmt_ctx->streams[i];
        AVRational fps = stream->avg_frame_rate;
        AVRational tbr = stream->r_frame_rate;
        AVRational tbn = stream->time_base;
        AVCodecParameters* codec_par = stream->codecpar;
        AVCodec* codec = avcodec_find_decoder(codec_par->codec_id);
        if (!codec) {
            fprintf(stderr, "Failed to find decoder for stream #%u\n", i);
            throw std::runtime_error("Failed to find decoder for stream.");
        }

        rgb_frame->format = AV_PIX_FMT_RGB24;
        rgb_frame->width = codec_par->width;
        rgb_frame->height = codec_par->height;
        ret = av_frame_get_buffer(rgb_frame, 0);
        if (ret < 0) {
            fprintf(stderr, "Failed to allocate buffer for the RGB frame\n");
            av_frame_free(&rgb_frame);
            throw std::runtime_error("Failed to allocate buffer for the RGB frame.");
        }        

        // Allocate memory to hold the context for decoding process
        AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx) {
            fprintf(stderr, "Failed to allocate the decoder context for stream #%u\n", i);
            throw std::runtime_error("Failed to allocate the decoder context for stream.");
        }
        // Fill codec context with codec parameters
        int ret = avcodec_parameters_to_context(codec_ctx, codec_par);
        if (ret < 0) {
            fprintf(stderr, "Failed to copy decoder parameters to input decoder context "
                    "for stream #%u\n", i);
            throw std::runtime_error("Failed to copy decoder parameters to input decoder context.");
        }
        
        if (codec_par->codec_type == AVMEDIA_TYPE_VIDEO
                || codec_par->codec_type == AVMEDIA_TYPE_AUDIO) {
            if (codec_ctx->codec_type == AVMEDIA_TYPE_VIDEO
                    || codec_ctx->codec_type == AVMEDIA_TYPE_AUDIO) {
                // Open the codec
                ret = avcodec_open2(codec_ctx, codec, nullptr);
                if (ret < 0) {
                    fprintf(stderr, "Failed to open decoder for stream #%u\n", i);
                    throw std::runtime_error("Failed to open decoder for stream.");
                } 
            }
        } 
        
        // Decode the frames in the media container
        std::vector<at::Tensor> one_stream;
        try {
            // Iterate over all frames in the stream
            while (av_read_frame(fmt_ctx, pkt) >= 0) {
                if (pkt->stream_index == i) {  
                    // Send raw data packet (compressed frame) to the decoder through the codec context
                    int ret = avcodec_send_packet(codec_ctx, pkt);
                    if (ret < 0) {
                        fprintf(stderr, "Error sending packet to decoder: %s\n");
                        throw std::runtime_error("Error sending packet to decoder.");
                    }
                    // Receive raw data frame (uncompressed frame) from the decoder through the codec context
                    while (ret >= 0) {
                        ret = avcodec_receive_frame(codec_ctx, frame);
                        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                            break;  // Need more input or decoding finished
                        } else if (ret < 0) {
                            fprintf(stderr, "Error receiving frame from decoder: %s\n");
                            throw std::runtime_error("Error receiving frame from decoder.");
                        }
                        
                        // Save the frame in a tensor
                        if (codec_par->codec_type == AVMEDIA_TYPE_VIDEO) {
                            // Convert to AV_PIX_FMT_RGB24 to guarantee 3 color channels
                            SwsContext *sws_ctx = sws_getContext(frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
                                                                    frame->width, frame->height, AV_PIX_FMT_RGB24,
                                                                    SWS_BILINEAR, nullptr, nullptr, nullptr);
                            if (!sws_ctx) {
                                fprintf(stderr, "Failed to create the conversion context\n");
                                throw std::runtime_error("Failed to create the conversion context.");
                            }
                            rgb_frame->format = AV_PIX_FMT_RGB24;
                            rgb_frame->width = frame->width;
                            rgb_frame->height = frame->height;
                            ret = av_frame_get_buffer(rgb_frame, 0);
                            if (ret < 0) {
                                fprintf(stderr, "Failed to allocate buffer for the RGB frame\n");
                                av_frame_free(&rgb_frame);
                                throw std::runtime_error("Failed to allocate buffer for the RGB frame.");
                            }  
                            ret = sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
                                            rgb_frame->data, rgb_frame->linesize);
                            if (ret < 0) {
                                fprintf(stderr, "Failed to convert the frame to RGB\n");
                                throw std::runtime_error("Failed to convert the frame to RGB.");
                            }
                            //int numBytes = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, frame->width, frame->height, 1);
                            int channels = 3;        
                            at::Tensor one_frame = at::empty({rgb_frame->height, rgb_frame->width, channels},
                            at::dtype(at::kByte).device(at::kCPU).pinned_memory(opts_.pin_memory()));
                            writable_memory_span frame_bits = get_raw_mutable_storage(one_frame);
                            auto frame_data = reinterpret_cast<uint8_t*>(frame_bits.data());
                            int row_size = rgb_frame->linesize[0];
                            unsigned char *buf = rgb_frame->data[0];  

                            for (int i = 0; i < rgb_frame->height; ++i) {
                                memcpy(frame_data + i * row_size, buf + i * row_size, row_size);
                            }                     
                            one_stream.push_back(std::move(one_frame)); 
                            sws_freeContext(sws_ctx);

                        } else if (codec_par->codec_type == AVMEDIA_TYPE_AUDIO) {
                            
                            int channels = av_get_channel_layout_nb_channels(frame->channel_layout);
                            //int channels = frame->channels;
                            //int frameSize = av_get_bytes_per_sample(static_cast<AVSampleFormat>(frame->format)) * channels * frame->nb_samples;
                            at::Tensor one_frame = at::empty({channels, frame->nb_samples},
                            at::dtype(at::kByte).device(at::kCPU).pinned_memory(opts_.pin_memory()));
                            writable_memory_span frame_bits = get_raw_mutable_storage(one_frame);
                            auto frame_data = reinterpret_cast<uint8_t*>(frame_bits.data());
                            int row_size = frame->linesize[0];

                            for (int c = 0; c < channels; ++c) {
                                memcpy(frame_data + c * row_size, frame->data[c], row_size);
                            }
                            one_stream.push_back(std::move(one_frame));
                            
                        } 
                        av_frame_unref(frame); // Unreference the old data of frame so it can be reused
                        av_frame_unref(rgb_frame);
                    }
                }
                av_packet_unref(pkt);   
            }
        } catch (const std::exception &e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            // TODO
        }
        all_streams.push_back(std::move(one_stream));
        avcodec_free_context(&codec_ctx);
    }
    av_packet_free(&pkt);
    av_frame_free(&frame);
    av_frame_free(&rgb_frame);
    if (avio_ctx) {
        av_freep(&avio_ctx->buffer);
        av_freep(&avio_ctx);
    }
    if (fmt_ctx) {
        avformat_free_context(fmt_ctx);
    }
    if(avio_ctx_buffer)
        av_freep(&avio_ctx_buffer);

    return all_streams;
}

int 
video_decoder::read_callback(void *opaque, uint8_t *buf, int buf_size) {
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
}
 
} // namespace fairseq2n
