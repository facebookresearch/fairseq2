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

int 
video_decoder::read_callback(void *opaque, uint8_t *buf, int buf_size) {
    video_decoder *decoder = static_cast<video_decoder*>(opaque);
    if (decoder == nullptr)
        return 0;
    long pos=0;
    long len=buf_size;
    if (pos < len) {
        auto available = std::min(int(len - pos), buf_size);
        memcpy(buf, buf+pos, available);
        pos += available;
        return available;
    }
    return 0;
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

    auto data_ptr = block.data();
    data output;
    

    std::cout << "check 1 " << std::endl;
    auto input_ctx = avformat_alloc_context();
    if (input_ctx == nullptr)
        throw std::runtime_error("Failed to allocate AVFormatContext.");
    std::cout << "check 2 " << std::endl;
    constexpr size_t io_buff_size = 96 * 1024;
    constexpr size_t io_pad_size = 64;
    //constexpr size_t log_buff_size = 1024;
    const size_t avio_ctx_buff_size = io_buff_size;
    uint8_t* avio_ctx_buff =
        (uint8_t*)av_malloc(avio_ctx_buff_size + io_pad_size);
    if (!avio_ctx_buff) {
        throw std::runtime_error("Failed to allocate AVIOContext buffer.");
    }
    std::cout << "check 3 " << std::endl;
    AVIOContext *avio_ctx = avio_alloc_context(
        reinterpret_cast<unsigned char*>(const_cast<std::byte*>(data_ptr)),
        avio_ctx_buff_size,                            
        0,                                           
        reinterpret_cast<void*>(const_cast<video_decoder*>(this)),                               
        &video_decoder::read_callback,                      
        nullptr,                                    
        nullptr                                      
    );
    std::cout << "check 4 " << std::endl;
    if (avio_ctx == nullptr) {
        avformat_free_context(input_ctx);
        throw std::runtime_error("Failed to allocate AVIOContext.");
    }
    input_ctx->pb = avio_ctx;
    input_ctx->flags |= AVFMT_FLAG_CUSTOM_IO;
    input_ctx->opaque = reinterpret_cast<void*>(const_cast<video_decoder*>(this));
    //input_ctx->interrupt_callback.callback = 
    input_ctx->interrupt_callback.opaque = reinterpret_cast<void*>(const_cast<video_decoder*>(this));
    input_ctx->flags |= AVFMT_FLAG_NONBLOCK;

    int result = avformat_open_input(&input_ctx, nullptr, nullptr, nullptr);
    std::cout << "check 5 " << std::endl;
    if (result < 0) {
        avformat_free_context(input_ctx);
        throw std::runtime_error("Failed to open input format context.");
    }
    std::cout << "check 6 " << std::endl;
    result = avformat_find_stream_info(input_ctx, nullptr);
    if (result < 0) {
        avformat_close_input(&input_ctx);
        throw std::runtime_error("Failed to find stream info.");
    }
    std::cout << "check 7 " << std::endl;
    av_dump_format(input_ctx, 0, nullptr, 0);
    avformat_close_input(&input_ctx);
    avformat_free_context(input_ctx);


    return output;

}  // namespace fairseq2n


}
