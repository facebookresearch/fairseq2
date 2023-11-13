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
    output.emplace("video", std::move(decoded_video));
    return output;

} 

std::vector<std::vector<uint8_t*>>
video_decoder::open_container(memory_block block)
{
    // Opens the media container and reads the metadata.

    auto data_ptr = reinterpret_cast<const uint8_t*>(block.data());
    //av_register_all();
    size_t data_size = block.size();
    int ret = 0;
    buffer_data bd = {0};   
    bd.ptr = data_ptr;
    bd.size = data_size;
    
    if (!(fmt_ctx_ = avformat_alloc_context())) {
        ret = AVERROR(ENOMEM);
        throw std::runtime_error("Failed to allocate AVFormatContext.");
    }
    
    avio_ctx_buffer_ = (uint8_t*)av_malloc(data_size + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!avio_ctx_buffer_) {
        ret = AVERROR(ENOMEM);
        clean();
        throw std::runtime_error("Failed to allocate AVIOContext buffer.");
    }

    avio_ctx_ = avio_alloc_context(avio_ctx_buffer_, data_size, 0, &bd, &video_decoder::read_callback, nullptr, nullptr);
    if (!avio_ctx_) {
        ret = AVERROR(ENOMEM);
        clean();
        throw std::runtime_error("Failed to allocate AVIOContext.");
    }
    fmt_ctx_->pb = avio_ctx_;
    fmt_ctx_->flags |= AVFMT_FLAG_CUSTOM_IO;
    fmt_ctx_->flags |= AVFMT_FLAG_NONBLOCK;
    // TODO: Determine the input format, currently causes seg fauit
    /*
    AVProbeData probe_data = {0};
    probe_data.buf = avio_ctx_buffer;
    probe_data.buf_size = data_size;
    probe_data.filename = "";  // Set to an empty string since we don't have a filename
    
    // Determine the input format
    fmt_ctx_->iformat = av_probe_input_format(&probe_data, 1);
    */
    ret = avformat_open_input(&fmt_ctx_, nullptr, nullptr, nullptr);
    if (ret < 0) {
        fprintf(stderr, "Could not open input\n");
        clean();
        throw std::runtime_error("Failed to open input.");
    }

    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (ret < 0) {
        fprintf(stderr, "Could not find stream information\n");
        clean();
        throw std::runtime_error("Failed to find stream information.");
    }

    av_dump_format(fmt_ctx_, 0, nullptr, 0);

    std::vector<std::vector<uint8_t*>> decoded_video = open_streams();

    clean();

    return decoded_video;

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

std::vector<std::vector<uint8_t*>>
video_decoder::open_streams()
{
    /* 
    Prepares for the decoding process by opening and initializing the decoders for 
    each stream in the AVFormatContext
    */
    std::vector<std::vector<uint8_t*>> allStreams;

    for (unsigned int i = 0; i < fmt_ctx_->nb_streams; i++) {
        //AVStream *stream = fmt_ctx_->streams[i];
        codec_par_ = fmt_ctx_->streams[i]->codecpar;
        codec_ = avcodec_find_decoder(codec_par_->codec_id);
        if (!codec_) {
            fprintf(stderr, "Failed to find decoder for stream #%u\n", i);
            throw std::runtime_error("Failed to find decoder for stream.");
        }
        codec_ctx_ = avcodec_alloc_context3(codec_);
        if (!codec_ctx_) {
            fprintf(stderr, "Failed to allocate the decoder context for stream #%u\n", i);
            throw std::runtime_error("Failed to allocate the decoder context for stream.");
        }
        int ret = avcodec_parameters_to_context(codec_ctx_, codec_par_);
        if (ret < 0) {
            fprintf(stderr, "Failed to copy decoder parameters to input decoder context "
                    "for stream #%u\n", i);
            throw std::runtime_error("Failed to copy decoder parameters to input decoder context.");
        }
        // Reencode video & audio and remux subtitles etc. 
        if (codec_par_->codec_type == AVMEDIA_TYPE_VIDEO
                || codec_par_->codec_type == AVMEDIA_TYPE_AUDIO) {
            if (codec_ctx_->codec_type == AVMEDIA_TYPE_VIDEO
                    || codec_ctx_->codec_type == AVMEDIA_TYPE_AUDIO) {
                // Open decoder 
                ret = avcodec_open2(codec_ctx_, codec_, nullptr);
                if (ret < 0) {
                    fprintf(stderr, "Failed to open decoder for stream #%u\n", i);
                    throw std::runtime_error("Failed to open decoder for stream.");
                } else {
                    if (codec_ctx_->codec_type == AVMEDIA_TYPE_VIDEO) {
                    /*
                    pImgConvertCtx = sws_getContext(codec_ctx_->width, codec_ctx_->height,
                            codec_ctx_->pix_fmt,
                            codec_ctx_->width, codec_ctx_->height,
                            AV_PIX_FMT_BGR24, // change later?
                            SWS_BICUBIC, NULL, NULL, NULL);
                    */
                    width_ = codec_ctx_->width;
                    height_ = codec_ctx_->height;
                }
                }
            }
        }
     
        std::vector<uint8_t*> stream = decode_frames(i);
        allStreams.push_back(std::move(stream));
    }
    return allStreams;
}

std::vector<uint8_t*> 
video_decoder::decode_frames(int stream_index)  
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

    try {
        // Iterate over all frames in the stream
        while (av_read_frame(fmt_ctx_, pkt) >= 0) {
            if (pkt->stream_index == stream_index) {  
                // Send packet to the decoder
                int ret = avcodec_send_packet(codec_ctx_, pkt);
                if (ret < 0) {
                    fprintf(stderr, "Error sending packet to decoder: %s\n");
                    throw std::runtime_error("Error sending packet to decoder.");
                }

                // Receive a decoded frame
                while (ret >= 0) {
                    ret = avcodec_receive_frame(codec_ctx_, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;  // Need more input or decoding finished
                    } else if (ret < 0) {
                        fprintf(stderr, "Error receiving frame from decoder: %s\n");
                        throw std::runtime_error("Error receiving frame from decoder.");
                    }

                    if (codec_par_->codec_type == AVMEDIA_TYPE_VIDEO) {
                        int frameSize = av_get_bytes_per_sample(AV_SAMPLE_FMT_S16) * av_frame_get_channels(frame) * frame->nb_samples;
                        outputArray = new uint8_t[frameSize];
                        int dataSize = av_samples_get_buffer_size(nullptr, av_frame_get_channels(frame), frame->nb_samples, AV_SAMPLE_FMT_S16, 0);
                        if (dataSize > 0) {
                            memcpy(outputArray, frame->data[0], dataSize);
                        }
                    } else if (codec_par_->codec_type == AVMEDIA_TYPE_AUDIO) {
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

void
video_decoder::clean() 
{
    if (fmt_ctx_)
        avformat_close_input(&fmt_ctx_);
    if (avio_ctx_) {
        av_freep(&avio_ctx_->buffer);
        avio_context_free(&avio_ctx_);
    }
    if(avio_ctx_buffer_)
        av_freep(&avio_ctx_buffer_);
    if (codec_ctx_)
        avcodec_free_context(&codec_ctx_);
}

 
} // namespace fairseq2n
