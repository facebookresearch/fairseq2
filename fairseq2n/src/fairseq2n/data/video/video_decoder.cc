// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/video/video_decoder.h"
#include "fairseq2n/data/video/detail/memory_buffer.h"
#include "fairseq2n/data/video/detail/utils.h"

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <iostream>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include "fairseq2n/exception.h"
#include "fairseq2n/float.h"
#include "fairseq2n/fmt.h"
#include "fairseq2n/memory.h"
#include "fairseq2n/data/detail/tensor_helpers.h"
#include "fairseq2n/detail/exception.h"

using namespace std;

#include "fairseq2n/exception.h"

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

    open_container(block);
    
    data output;
    return output;

} 

int
video_decoder::open_container(memory_block block) const
{
    // Opens the media container and reads the metadata.

    auto data_ptr = reinterpret_cast<const uint8_t*>(block.data());
    //av_register_all();
    AVFormatContext* fmt_ctx = nullptr;
    AVIOContext *avio_ctx = nullptr;
    size_t data_size = block.size();
    int ret = 0;
    buffer_data bd = {0};   
    bd.ptr = data_ptr;
    bd.size = data_size;

    if (!(fmt_ctx = avformat_alloc_context())) {
        ret = AVERROR(ENOMEM);
        throw std::runtime_error("Failed to allocate AVFormatContext.");
    }

    uint8_t *avio_ctx_buffer = (uint8_t*)av_malloc(data_size + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!avio_ctx_buffer) {
        ret = AVERROR(ENOMEM);
        avformat_close_input(&fmt_ctx);
        throw std::runtime_error("Failed to allocate AVIOContext buffer.");
    }

    avio_ctx = avio_alloc_context(avio_ctx_buffer, data_size, 0, &bd, &video_decoder::read_callback, nullptr, nullptr);
    if (!avio_ctx) {
        ret = AVERROR(ENOMEM);
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
    ret = avformat_open_input(&fmt_ctx, nullptr, nullptr, nullptr);
    if (ret < 0) {
        fprintf(stderr, "Could not open input\n");
        avformat_close_input(&fmt_ctx);
        av_freep(&avio_ctx->buffer);
        avio_context_free(&avio_ctx);
        throw std::runtime_error("Failed to open input.");
    }

    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        fprintf(stderr, "Could not find stream information\n");
        avformat_close_input(&fmt_ctx);
        av_freep(&avio_ctx->buffer);
        avio_context_free(&avio_ctx);
        throw std::runtime_error("Failed to find stream information.");
    }

    av_dump_format(fmt_ctx, 0, nullptr, 0);

    open_streams(fmt_ctx);

    avformat_close_input(&fmt_ctx);
    av_freep(&avio_ctx->buffer);
    avio_context_free(&avio_ctx);

    return 0;

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

int
video_decoder::open_streams(AVFormatContext* fmt_ctx) const
{
    /* 
    Prepares for the decoding process by opening and initializing the decoders for 
    each stream in the AVFormatContext
    */

    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        AVStream *stream = fmt_ctx->streams[i];
        AVCodecParameters *codecpar = stream->codecpar;
        AVCodec *codec = avcodec_find_decoder(codecpar->codec_id);
        if (!codec) {
            fprintf(stderr, "Failed to find decoder for stream #%u\n", i);
            throw std::runtime_error("Failed to find decoder for stream.");
        }
        AVCodecContext *codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx) {
            fprintf(stderr, "Failed to allocate the decoder context for stream #%u\n", i);
            throw std::runtime_error("Failed to allocate the decoder context for stream.");
        }
        int ret = avcodec_parameters_to_context(codec_ctx, codecpar);
        if (ret < 0) {
            fprintf(stderr, "Failed to copy decoder parameters to input decoder context "
                    "for stream #%u\n", i);
            throw std::runtime_error("Failed to copy decoder parameters to input decoder context.");
        }
        // Reencode video & audio and remux subtitles etc. 
        if (codecpar->codec_type == AVMEDIA_TYPE_VIDEO
                || codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            if (codec_ctx->codec_type == AVMEDIA_TYPE_VIDEO
                    || codec_ctx->codec_type == AVMEDIA_TYPE_AUDIO) {
                // Open decoder 
                ret = avcodec_open2(codec_ctx, codec, nullptr);
                if (ret < 0) {
                    fprintf(stderr, "Failed to open decoder for stream #%u\n", i);
                    throw std::runtime_error("Failed to open decoder for stream.");
                }
            }
        }
        // TODO: call decode_frames and pass i as stream_index
    }
    return 0;
}

int video_decoder::decode_frame(AVFormatContext* fmt_ctx, AVCodecContext *codec_ctx, int stream_index) const 
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

    try {
        while (av_read_frame(fmt_ctx, pkt) >= 0) {
            if (pkt->stream_index == stream_index) {  
                // Send packet to the decoder
                int ret = avcodec_send_packet(codec_ctx, pkt);
                if (ret < 0) {
                    fprintf(stderr, "Error sending packet to decoder: %s\n", av_err2str(ret));
                    throw std::runtime_error("Error sending packet to decoder.");
                }

                // Receive decoded frames
                while (ret >= 0) {
                    ret = avcodec_receive_frame(codec_ctx, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;  // Need more input or decoding finished
                    } else if (ret < 0) {
                        fprintf(stderr, "Error receiving frame from decoder: %s\n", av_err2str(ret));
                        throw std::runtime_error("Error receiving frame from decoder.");
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

    return 0; 
}

void
video_decoder::clean() const
{
// TODO
}

 
} // namespace fairseq2n
