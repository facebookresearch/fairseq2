// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/audio/audio_decoder.h"

#include <cstdint>
#include <exception>
#include <stdexcept>

#include <ATen/Functions.h>
#include <ATen/Storage.h>
#include <ATen/Tensor.h>

#include "fairseq2/native/exception.h"
#include "fairseq2/native/fmt.h"
#include "fairseq2/native/memory.h"
#include "fairseq2/native/span.h"
#include "fairseq2/native/data/audio/detail/sndfile.h"

using namespace fairseq2::detail;

namespace fairseq2 {

audio_decoder::audio_decoder(audio_decoder_options opts)
  : opts_{opts}
{
    at::ScalarType dtype = opts_.dtype().value_or(at::kFloat);
    if (dtype != at::kFloat && dtype != at::kInt)
        throw not_supported_error{
            "`audio_decoder` supports only `torch.float` and `torch.int` data types."};
}

data
audio_decoder::operator()(data &&d) const
{
    if (!d.is_memory_block())
        throw std::invalid_argument{
            fmt::format("The input data must be of type `memory_block`, but is of type `{}` instead.", d.type())};

    const memory_block &block = d.as_memory_block();
    if (block.empty())
        throw std::invalid_argument{
            "The input memory block has zero length and cannot be decoded as audio."};

    sndfile file{};
    try {
        file = sndfile::from_memory(block);
    } catch (const std::invalid_argument &) {
        std::throw_with_nested(std::invalid_argument{
            "The input audio cannot be decoded. See nested exception for details."});
    } catch (const std::runtime_error &) {
        std::throw_with_nested(std::runtime_error{
            "The input audio cannot be decoded. See nested exception for details."});
    }

    at::ScalarType dtype = opts_.dtype().value_or(at::kFloat);

    at::Tensor tensor = at::empty({file.num_frames(), file.num_channels()},
        at::dtype(dtype).device(at::kCPU).pinned_memory(opts_.pin_memory()));

    const at::Storage &storage = tensor.storage();

    writable_memory_span tensor_bits{storage.unsafe_data<std::byte>(), storage.nbytes()};

    switch (dtype) {
    case at::kFloat: {
        span tensor_data = cast<float32>(tensor_bits);

        file.decode_into(tensor_data);

        break;
    }
    case at::kInt: {
        span tensor_data = cast<std::int32_t>(tensor_bits);

        file.decode_into(tensor_data);

        break;
    }
    default:
        throw internal_error{
            "`audio_decoder` uses an unsupported data type. Please file a bug report."};
    };

    at::Device device = opts_.device().value_or(at::kCPU);
    if (device != at::kCPU)
        tensor = tensor.to(device);

    // Pack audio, sample_rate, and format as output.
    data_dict output{{"sample_rate", file.sample_rate()}, {"format", file.format()}};

    output["audio"] = std::move(tensor);

    return output;
}

}  // namespace fairseq2
