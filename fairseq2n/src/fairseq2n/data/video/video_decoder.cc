// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/video/video_decoder.h"
#include "fairseq2n/data/video/detail/memory_buffer.h"

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <iostream>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavformat/avio.h>
    #include <libavutil/avutil.h>
}

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

struct BufferData {
    const uint8_t *ptr; 
    size_t size;        
};

int 
video_decoder::read_callback(void *opaque, uint8_t *buf, int buf_size) {
    
    BufferData *bd = static_cast<BufferData *>(opaque);
    buf_size = std::min(buf_size, static_cast<int>(bd->size));

    if (buf_size <= 0)
        return AVERROR_EOF;

    // Copy internal buffer data to buf
    memcpy(buf, bd->ptr, buf_size);
    bd->ptr += buf_size;
    bd->size -= buf_size;
    return buf_size;
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

    auto data_ptr = reinterpret_cast<const uint8_t*>(block.data());
    //av_register_all();
    AVFormatContext* fmt_ctx = nullptr;
    AVIOContext *avio_ctx = nullptr;
    size_t data_size = block.size();
    int ret = 0;
    BufferData bd = {0};
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

    AVProbeData probe_data = {0};
    probe_data.buf = const_cast<uint8_t *>(data_ptr);
    probe_data.buf_size = data_size + data_size;
    probe_data.filename = "";  

    // Determine the input format
    fmt_ctx->iformat = av_probe_input_format(&probe_data, 1);

    ret = avformat_open_input(&fmt_ctx, nullptr, fmt_ctx->iformat, nullptr);
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
    avformat_close_input(&fmt_ctx);
    av_freep(&avio_ctx->buffer);
    avio_context_free(&avio_ctx);

    data output;
    return output;

}  // namespace fairseq2n


}
