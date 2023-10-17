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

    auto data_ptr = block.data();
    auto data_len = block.size();

    struct Reader {
    png_const_bytep ptr;
    png_size_t count;
    } reader;

    reader.ptr = png_const_bytep(data_ptr) + 8;
    reader.count = data_len - 8;
    
    // Define custom read function
    auto read_callback = [](png_structp png_ptr,
                          png_bytep output,
                          png_size_t bytes) {
    auto reader = static_cast<Reader*>(png_get_io_ptr(png_ptr));
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
    int channels = png_get_channels(png_ptr, info_ptr);

    // Allocate memory for image data
    int rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    png_bytep* row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*) malloc(rowbytes);
    }

    // Read image data row by row
    png_read_image(png_ptr, row_pointers);

    // Specify tensor data type
    at::ScalarType dtype = opts_.maybe_dtype().value_or(at::kFloat);

    // Copy image data into tensor object
    at::Tensor image = at::empty({height, width, 3}, at::dtype(dtype).device(at::kCPU).pinned_memory(opts_.pin_memory()));
    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 3]);
            image[y][x][0] = px[0];
            image[y][x][1] = px[1];
            image[y][x][2] = px[2];
        }
    }

    // Free memory for image data 
    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
 
    // Move tensor to specified device
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