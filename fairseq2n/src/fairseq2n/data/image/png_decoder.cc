// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/image/png_decoder.h"

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <png.h>
#include <iostream>

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
    
png_decoder::png_decoder(png_decoder_options opts)
  : opts_{opts}
{
    at::ScalarType dtype = opts_.maybe_dtype().value_or(at::kFloat);
    if (dtype != at::kFloat && dtype != at::kByte)
        throw_<not_supported_error>(
            "`png_decoder` supports only `torch.float32` and `torch.uint8` data types.");
}

data
png_decoder::operator()(data &&d) const
{
    if (!d.is_memory_block())
        throw_<std::invalid_argument>(
            "The input data must be of type `memory_block`, but is of type `{}` instead.", d.type());

    const memory_block &block = d.as_memory_block();
    if (block.empty())
        throw_<std::invalid_argument>(
            "The input memory block has zero length and cannot be decoded as png.");

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        throw_<internal_error>("Failed to create PNG read struct.");
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        throw_<internal_error>("Failed to create PNG info struct.");
    }

    // Set up error handling.
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        throw std::runtime_error("Error reading PNG image from memory");
    }

    auto datap = block.data();
    auto datap_len = block.size();

    struct Reader {
    png_const_bytep ptr;
    png_size_t count;
    } reader;

    reader.ptr = png_const_bytep(datap) + 8;
    reader.count = datap_len - 8;

    auto read_callback = [](png_structp png_ptr,
                          png_bytep output,
                          png_size_t bytes) {
    auto reader = static_cast<Reader*>(png_get_io_ptr(png_ptr));
    TORCH_CHECK(
        reader->count >= bytes,
        "Out of bound read in png_decoder. Probably, the input image is corrupted");
    std::copy(reader->ptr, reader->ptr + bytes, output);
    reader->ptr += bytes;
    reader->count -= bytes;
    };
    png_set_sig_bytes(png_ptr, 8);
    png_set_read_fn(png_ptr, &reader, read_callback);
    png_read_info(png_ptr, info_ptr);

    png_uint_32 width = png_get_image_width(png_ptr, info_ptr);
    png_uint_32 height = png_get_image_height(png_ptr, info_ptr);
    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);

    
    // temporary check to confirm image is being read
    std::cout << "img width:" << width << std::endl;
    
    /*
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
        {// todo
        }

    */

    

}
};