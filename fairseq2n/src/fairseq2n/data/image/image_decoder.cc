// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/image/image_decoder.h"

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <png.h>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include "fairseq2n/exception.h"
#include "fairseq2n/float.h"
#include "fairseq2n/fmt.h"
#include "fairseq2n/memory.h"
#include "fairseq2n/span.h"
#include "fairseq2n/data/audio/detail/sndfile.h"
#include "fairseq2n/data/detail/tensor_helpers.h"
#include "fairseq2n/detail/exception.h"

using namespace fairseq2n::detail;

namespace fairseq2n {
    
image_decoder::image_decoder(image_decoder_options opts)
  : opts_{opts}
{
    at::ScalarType dtype = opts_.maybe_dtype().value_or(at::kFloat);
    if (dtype != at::kFloat && dtype != at::kByte)
        throw_<not_supported_error>(
            "`image_decoder` supports only `torch.float32` and `torch.uint8` data types.");
}

data
image_decoder::operator()(data &&d) const
{
    if (!d.is_memory_block())
        throw_<std::invalid_argument>(
            "The input data must be of type `memory_block`, but is of type `{}` instead.", d.type());

    const memory_block &block = d.as_memory_block();
    if (block.empty())
        throw_<std::invalid_argument>(
            "The input memory block has zero length and cannot be decoded as audio.");

    png_bytep buffer = reinterpret_cast<png_bytep>(const_cast<void*>(static_cast<const void*>(block.data())));

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        // Handle error
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        // Handle error
    }
    
    png_rw_ptr read_fn = [](png_structp png_ptr, png_bytep data, png_size_t length) {
        png_bytep& buffer = *reinterpret_cast<png_bytep*>(png_get_io_ptr(png_ptr));
        memcpy(data, buffer, length);
        buffer += length;
    };
    png_set_read_fn(png_ptr, &buffer, read_fn);

    png_read_info(png_ptr, info_ptr);
    png_uint_32 width = png_get_image_width(png_ptr, info_ptr);
    png_uint_32 height = png_get_image_height(png_ptr, info_ptr);
    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);

   

    at::ScalarType dtype = opts_.maybe_dtype().value_or(at::kFloat);

    at::Tensor rgb = at::empty({width, height},
        at::dtype(dtype).device(at::kCPU).pinned_memory(opts_.pin_memory()));

    writable_memory_span rgb_bits = get_raw_mutable_storage(rgb);

    switch (dtype) {
    case at::kFloat: {
        span waveform_data = cast<float32>(rgb_bits);

        // todo

        break;
    }
    case at::kByte: {
        span waveform_data = cast<std::uint8_t>(rgb_bits);

        // todo

        break;
    }
    default:
        throw_<internal_error>(
            "`image_decoder` uses an unsupported data type. Please file a bug report.");
    };

    at::Device device = opts_.maybe_device().value_or(at::kCPU);
    if (device != at::kCPU)
        // todo

}
};