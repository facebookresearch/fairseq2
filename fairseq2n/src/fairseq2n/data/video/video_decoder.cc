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

using namespace fairseq2n::detail;

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

    std::vector<std::vector<uint8_t*>> decoded_video = open_container(block);
    
    data_dict output;
    //output.emplace("video", std::move(decoded_video));
    return output;

} 

std::vector<std::vector<uint8_t*>>
video_decoder::open_container(memory_block block) const
{
    // Opens the media container and reads the metadata.
    
    auto data_ptr = reinterpret_cast<const uint8_t*>(block.data());
    //av_register_all();
    size_t data_size = block.size();
    buffer_data bd = {0};   
    bd.ptr = data_ptr;
    bd.size = data_size;
    avformat_resources format_resources(data_size, bd);
    AVFormatContext* fmt_ctx = format_resources.get_fmt_ctx();
    AVIOContext* avio_ctx = format_resources.get_avio_ctx();
    int ret = 0;
    
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
    ret = avformat_open_input(&fmt_ctx, nullptr, nullptr, nullptr);
    if (ret < 0) {
        fprintf(stderr, "Could not open input\n");
        throw std::runtime_error("Failed to open input.");
    }

    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        fprintf(stderr, "Could not find stream information\n");
        throw std::runtime_error("Failed to find stream information.");
    }

    av_dump_format(fmt_ctx, 0, nullptr, 0);

    std::vector<std::vector<uint8_t*>> decoded_video = open_streams(format_resources);



    return decoded_video;

}

std::vector<std::vector<uint8_t*>>
video_decoder::open_streams(avformat_resources format_resources) const
{
    /* 
    Prepares for the decoding process by opening and initializing the decoders for 
    each stream in the AVFormatContext
    */

    AVFormatContext* fmt_ctx = format_resources.get_fmt_ctx();
    std::vector<std::vector<uint8_t*>> allStreams;

    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        //AVStream *stream = fmt_ctx->streams[i];
        AVCodecParameters* codec_par = fmt_ctx->streams[i]->codecpar;
        AVCodec* codec = avcodec_find_decoder(codec_par->codec_id);
        if (!codec) {
            fprintf(stderr, "Failed to find decoder for stream #%u\n", i);
            throw std::runtime_error("Failed to find decoder for stream.");
        }
        avcodec_resources codec_resources(codec);
        AVCodecContext* codec_ctx = codec_resources.get_codec_ctx();
        
        codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx) {
            fprintf(stderr, "Failed to allocate the decoder context for stream #%u\n", i);
            throw std::runtime_error("Failed to allocate the decoder context for stream.");
        }
        int ret = avcodec_parameters_to_context(codec_ctx, codec_par);
        if (ret < 0) {
            fprintf(stderr, "Failed to copy decoder parameters to input decoder context "
                    "for stream #%u\n", i);
            throw std::runtime_error("Failed to copy decoder parameters to input decoder context.");
        }
        // Reencode video & audio and remux subtitles etc. 
        if (codec_par->codec_type == AVMEDIA_TYPE_VIDEO
                || codec_par->codec_type == AVMEDIA_TYPE_AUDIO) {
            if (codec_ctx->codec_type == AVMEDIA_TYPE_VIDEO
                    || codec_ctx->codec_type == AVMEDIA_TYPE_AUDIO) {
                // Open decoder 
                ret = avcodec_open2(codec_ctx, codec, nullptr);
                if (ret < 0) {
                    fprintf(stderr, "Failed to open decoder for stream #%u\n", i);
                    throw std::runtime_error("Failed to open decoder for stream.");
                } else {
                    if (codec_ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
                    /*
                    pImgConvertCtx = sws_getContext(codec_ctx_->width, codec_ctx_->height,
                            codec_ctx_->pix_fmt,
                            codec_ctx_->width, codec_ctx_->height,
                            AV_PIX_FMT_BGR24, // change later?
                            SWS_BICUBIC, NULL, NULL, NULL);
                    */
                    //width_ = codec_ctx->width;
                    //height_ = codec_ctx->height;
                }
                }
            }
        }
     
        std::vector<uint8_t*> stream = decode_frames(i, format_resources, codec_resources);
        allStreams.push_back(std::move(stream));
    }
    return allStreams;
}

std::vector<uint8_t*> 
video_decoder::decode_frames(int stream_index, avformat_resources format_resources, avcodec_resources codec_resources) const
{
    /*
    Decodes the frames in the media container and returns a tensor of the decoded frames
    */
     
    AVPacket *pkt = av_packet_alloc();
    if (!pkt) {
        fprintf(stderr, "Failed to allocate the packet\n");
        throw std::runtime_error("Failed to allocate the packet.");
    }

    AVFrame *frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Failed to allocate the frame\n");
        throw std::runtime_error("Failed to allocate the frame.");
    }

    std::vector<uint8_t*> frameData; 
    uint8_t* outputArray = nullptr;
    AVCodecContext* codec_ctx = codec_resources.get_codec_ctx();
    AVFormatContext* fmt_ctx = format_resources.get_fmt_ctx();
    AVIOContext* avio_ctx = format_resources.get_avio_ctx();
    AVCodecParameters* codec_par = fmt_ctx->streams[stream_index]->codecpar;

    try {
        // Iterate over all frames in the stream
        while (av_read_frame(fmt_ctx, pkt) >= 0) {
            if (pkt->stream_index == stream_index) {  
                // Send packet to the decoder
                int ret = avcodec_send_packet(codec_ctx, pkt);
                if (ret < 0) {
                    fprintf(stderr, "Error sending packet to decoder: %s\n");
                    throw std::runtime_error("Error sending packet to decoder.");
                }

                // Receive a decoded frame
                while (ret >= 0) {
                    ret = avcodec_receive_frame(codec_ctx, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;  // Need more input or decoding finished
                    } else if (ret < 0) {
                        fprintf(stderr, "Error receiving frame from decoder: %s\n");
                        throw std::runtime_error("Error receiving frame from decoder.");
                    }

                    if (codec_par->codec_type == AVMEDIA_TYPE_VIDEO) {
                        int frameSize = av_get_bytes_per_sample(AV_SAMPLE_FMT_S16) * av_frame_get_channels(frame) * frame->nb_samples;
                        outputArray = new uint8_t[frameSize];
                        int dataSize = av_samples_get_buffer_size(nullptr, av_frame_get_channels(frame), frame->nb_samples, AV_SAMPLE_FMT_S16, 0);
                        if (dataSize > 0) {
                            memcpy(outputArray, frame->data[0], dataSize);
                        }
                    } else if (codec_par->codec_type == AVMEDIA_TYPE_AUDIO) {
                        uint8_t* audio_data = frame->data[0];
                        int audio_data_size = av_get_bytes_per_sample(static_cast<AVSampleFormat>(frame->format)) * frame->nb_samples;
                        outputArray = new uint8_t[audio_data_size];
                        memcpy(outputArray, audio_data, audio_data_size);
                    }
                    if (outputArray) {
                        frameData.push_back(outputArray);
                    }


                    // TODO: process decoded frame
                }
            }

            av_packet_unref(pkt);
        }
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        // TODO
    }

    av_packet_free(&pkt);
    av_frame_free(&frame);

    return frameData; 
}

int 
video_decoder::read_callback(void *opaque, uint8_t *buf, int buf_size) {
    // Read up to buf_size bytes from the resource accessed by the AVIOContext object
    // Used by ffmpeg to read from memory buffer
    buffer_data *bd = static_cast<buffer_data *>(opaque);
    buf_size = std::min(buf_size, static_cast<int>(bd->size));
    if (buf_size <= 0)
        return AVERROR_EOF;
    memcpy(buf, bd->ptr, buf_size);
    bd->ptr += buf_size;
    bd->size -= buf_size;
    return buf_size;
}
 
} // namespace fairseq2n
