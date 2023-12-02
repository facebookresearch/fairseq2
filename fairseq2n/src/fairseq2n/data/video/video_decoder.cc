// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/video/video_decoder.h"

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
using namespace fairseq2n::detail;

namespace fairseq2n {

video_decoder::video_decoder(video_decoder_options opts, bool pin_memory)
    : opts_{opts}
{
    /*
    dtype is used to determine the type of the output tensor for raw frame data only,
    which is usually stored as unsigned 8-bit or 10-bit integers corresponding to Kbyte
    and kShort in Pytorch. 
    */
    at::ScalarType dtype = opts_.maybe_dtype().value_or(at::kByte);
    if (dtype != at::kByte && dtype != at::kShort)
        throw not_supported_error(
            "`video_decoder` supports only `torch.int16` and `torch.uint8` data types.");
    opts_.pin_memory(pin_memory);
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

    ffmpeg_decoder decoder(opts_);

    data_dict decoded_video = decoder.open_container(block);

    data_dict output;
    output.emplace("video", std::move(decoded_video));
    return output;
} 

} // namespace fairseq2n
