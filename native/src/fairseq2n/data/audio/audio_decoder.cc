// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/audio/audio_decoder.h"

#include <cstdint>
#include <exception>
#include <stdexcept>

#include <ATen/Functions.h>
#include <ATen/ScalarType.h>
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

audio_decoder::audio_decoder(audio_decoder_options opts)
  : opts_{opts}
{
    at::ScalarType dtype = opts_.maybe_dtype().value_or(at::kFloat);

    if (!at::isFloatingType(dtype) && !at::isIntegralType(dtype, /*includeBool=*/false))
        throw_<not_supported_error>(
            "`audio_decoder` supports only integral and floating-point types.");
}

data
audio_decoder::operator()(data &&d) const
{
    if (!d.is_memory_block())
        throw_<std::invalid_argument>(
            "The input data must be of type `memory_block`, but is of type `{}` instead.", d.type());

    const memory_block &block = d.as_memory_block();
    if (block.empty())
        throw_<std::invalid_argument>(
            "The input memory block has zero length and cannot be decoded as audio.");

    sndfile file{};
    try {
        file = sndfile::from_memory(block);
    } catch (const std::invalid_argument &) {
        throw_with_nested<std::invalid_argument>(
            "The input audio cannot be decoded. See nested exception for details.");
    } catch (const std::runtime_error &) {
        throw_with_nested<std::invalid_argument>(
            "The input audio cannot be decoded. See nested exception for details.");
    }

    at::ScalarType dtype = opts_.maybe_dtype().value_or(at::kFloat);

    at::ScalarType decode_dtype{};

    if (at::isFloatingType(dtype))
        decode_dtype = at::kFloat;
    else if (dtype == at::kShort)
        decode_dtype = at::kShort;
    else if (at::isIntegralType(dtype, /*includeBool=*/false))
        decode_dtype = at::kInt;
    else
        throw_<internal_error>(
            "`audio_decoder` uses an unsupported data type. Please file a bug report.");

    at::Tensor waveform = at::empty({file.num_frames(), file.num_channels()},
        at::dtype(decode_dtype).device(at::kCPU).pinned_memory(opts_.pin_memory()));

    writable_memory_span waveform_bits = get_raw_mutable_storage(waveform);

    switch (decode_dtype) {
    case at::kFloat: {
        span waveform_data = cast<float32>(waveform_bits);

        file.decode_into(waveform_data);

        break;
    }
    case at::kShort: {
        span waveform_data = cast<std::int16_t>(waveform_bits);

        file.decode_into(waveform_data);

        break;
    }
    case at::kInt: {
        span waveform_data = cast<std::int32_t>(waveform_bits);

        file.decode_into(waveform_data);

        break;
    }
    default:
        throw_<internal_error>(
            "`audio_decoder` uses an unsupported data type. Please file a bug report.");
    };

    if (file.num_channels() == 1 && !opts_.keepdim())
        waveform = waveform.squeeze(-1);

    waveform = waveform.to(dtype);

    at::Device device = opts_.maybe_device().value_or(at::kCPU);
    if (device != at::kCPU)
        waveform = waveform.to(device);

    // Pack audio (i.e. waveform), sample_rate, and format as output.
    data_dict output{
        {"sample_rate", static_cast<float32>(file.sample_rate())}, {"format", file.format()}};

    output.emplace("waveform", std::move(waveform));

    return output;
}

}  // namespace fairseq2n
