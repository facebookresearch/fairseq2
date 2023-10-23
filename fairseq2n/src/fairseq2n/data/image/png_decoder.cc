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
#include "fairseq2n/data/detail/tensor_helpers.h"
#include "fairseq2n/detail/exception.h"

using namespace fairseq2n::detail;

namespace fairseq2n {
    
png_decoder::png_decoder(png_decoder_options opts)
  : opts_{opts}
{}

bool 
png_decoder::is_little_endian() {
  uint32_t x = 1;
  return (*reinterpret_cast<uint8_t*>(&x) == 1);
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
  
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (png_ptr == nullptr) {
        throw_<internal_error>("Failed to create PNG read struct.");
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == nullptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        throw_<internal_error>("Failed to create PNG info struct.");
    }

    auto data_ptr = png_const_bytep(block.data());
    auto data_len = block.size();

    struct Reader {
    png_const_bytep ptr;
    png_size_t count;
    Reader(png_const_bytep p, png_size_t c) : ptr(p), count(c) {}
    };
    Reader reader(data_ptr + 8, data_len - 8);
    
    auto read_callback = [](png_structp png_ptr2,
                          png_bytep output,
                          png_size_t bytes) {
    auto reader = static_cast<Reader*>(png_get_io_ptr(png_ptr2));
    std::copy(reader->ptr, reader->ptr + bytes, output);
    reader->ptr += bytes;
    reader->count -= bytes;
    };

    png_set_sig_bytes(png_ptr, 8);
    png_set_read_fn(png_ptr, &reader, read_callback);
    png_read_info(png_ptr, info_ptr);

    png_uint_32 width=0, height=0;
    int bit_depth=0, color_type=0;
    int interlace_type=0;
    auto retval = png_get_IHDR(
        png_ptr,
        info_ptr,
        &width,
        &height,
        &bit_depth,
        &color_type,
        &interlace_type,
        nullptr,
        nullptr);

    if (retval != 1) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        throw_<std::invalid_argument>("Could not read image metadata from content.");
    }

    if (is_little_endian()) {
      png_set_swap(png_ptr);
    }
    int channels = png_get_channels(png_ptr, info_ptr);

    at::ScalarType dtype = bit_depth <= 8 ? at::kByte : at::kShort;
    at::Tensor image = at::empty({height, width, channels}, at::dtype(dtype).device(at::kCPU).pinned_memory(opts_.pin_memory()));
    
    size_t rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    writable_memory_span image_bits = get_raw_mutable_storage(image);
    auto image_data = reinterpret_cast<png_bytep>(image_bits.data());
    
    // Read image data into tensor
    for (png_uint_32 i = 0; i < height; ++i) {
        png_read_row(png_ptr, image_data, nullptr);
        image_data += rowbytes;
    }

    at::Device device = opts_.maybe_device().value_or(at::kCPU);
    if (device != at::kCPU)
        image = image.to(device);

    // Pack png data and format as output.
    data_dict output{
        {"bit_depth", static_cast<float32>(bit_depth)}, {"color_type", static_cast<float32>(color_type)}, 
        {"channels", static_cast<float32>(channels)}, {"height", static_cast<float32>(height)}, 
        {"width", static_cast<float32>(width)}};

    output.emplace("image", std::move(image));
    
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    return output;
}
};
