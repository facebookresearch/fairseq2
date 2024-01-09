// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/audio/waveform_to_fbank_converter.h"

#include <limits>
#include <tuple>
#include <utility>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <kaldi-native-fbank/csrc/feature-fbank.h>

#include "fairseq2n/fmt.h"
#include "fairseq2n/detail/exception.h"
#include "fairseq2n/data/audio/detail/kaldi_fbank.h"

using namespace fairseq2n::detail;

namespace fairseq2n {

waveform_to_fbank_converter::waveform_to_fbank_converter(fbank_options opts) noexcept
  : opts_{opts}
{}

waveform_to_fbank_converter::~waveform_to_fbank_converter() = default;

data
waveform_to_fbank_converter::operator()(data &&d) const
{
    if (!d.is_dict())
        throw_<std::invalid_argument>(
            "The input data must be of type `dict` containing a waveform tensor and its sample rate, but is of type `{}` instead.", d.type());

    data_dict &dict = d.as_dict();

    float32 sample_rate = find_sample_rate(dict);
    if (computer_ == nullptr) {
        // Any sample rate below 100 causes Kaldi to underflow.
        if (sample_rate < 100.0F)
            throw_<std::invalid_argument>(
                "The input sample rate must be greater than or equal to 100, but is {:G} instead.", sample_rate);

        std::lock_guard<std::mutex> init_guard{init_mutex_};

        if (computer_ == nullptr)
            init_computer(sample_rate);
    } else {
        if (!are_close(computer_->sample_rate(), sample_rate))
            throw_<std::invalid_argument>(
                "The input waveform must have a sample rate of {}, but has a sample rate of {:G} instead.", computer_->sample_rate(), sample_rate);
    }

    at::Tensor waveform = find_waveform(dict);

    if (opts_.channel_last())
        waveform = waveform.transpose(0, 1);

    waveform = waveform.to(
        at::kCPU, at::kFloat, /*non_blocking=*/false, /*copy=*/false, at::MemoryFormat::Contiguous);

    if (!are_close(opts_.waveform_scale(), 1.0F))
        waveform = waveform.multiply(opts_.waveform_scale());

    at::Tensor fbank = computer_->compute(waveform, opts_.pin_memory());

    if (opts_.standardize()) {
        at::Tensor stdev{}, mean{};

        std::tie(stdev, mean) = at::std_mean(fbank, /*dim=*/0);

        fbank = fbank.subtract(mean).divide(stdev);
    }

    // If no device is specified, we fallback to the device of the waveform
    // instead of the default floating-point type.
    at::Device device = opts_.maybe_device().value_or(waveform.device());

    at::ScalarType dtype = opts_.maybe_dtype().value_or(at::kFloat);

    fbank = fbank.to(device, dtype);

    // Ensure that we return the sample rate always in floating-point.
    dict["sample_rate"] = static_cast<float64>(sample_rate);

    if (!opts_.keep_waveform())
        dict.erase("waveform");

    dict.emplace("fbank", std::move(fbank));

    return std::move(d);
}

at::Tensor &
waveform_to_fbank_converter::find_waveform(data_dict &dict)
{
    auto pos = dict.find("waveform");
    if (pos == dict.end())
        throw_<std::invalid_argument>(
            "The input dictionary must contain the waveform under a key named `waveform`, but does not contain such key.");

    data &element = pos->second;
    if (!element.is_tensor())
        throw_<std::invalid_argument>(
            "The input waveform must be of type `torch.Tensor`, but is of type `{}` instead.", element.type());

    at::Tensor &waveform = element.as_tensor();

    if (waveform.dim() != 2)
        throw_<std::invalid_argument>(
            "The input waveform must be two dimensional, but has {} dimension(s) instead.", waveform.dim());

    return waveform;
}

float32
waveform_to_fbank_converter::find_sample_rate(const data_dict &dict)
{
    auto pos = dict.find("sample_rate");
    if (pos == dict.end())
        throw_<std::invalid_argument>(
            "The input dictionary must contain the waveform sample rate under a key named `sample_rate`, but does not contain such key.");

    const data &element = pos->second;
    if (!element.is_float() && !element.is_int())
        throw_<std::invalid_argument>(
            "The input sample rate must be of type `float` or `int`, but is of type `{}` instead.", element.type());

    float64 fp64_sample_rate{};
    if (element.is_float())
        fp64_sample_rate = element.as_float();
    else
        fp64_sample_rate = static_cast<float64>(element.as_int());

    float32 sample_rate{};
    if (!maybe_narrow(fp64_sample_rate, sample_rate))
        throw_<std::invalid_argument>(
            "The input sample rate must be representable in single precision (32-bit), but is {:G} instead.", fp64_sample_rate);

    return sample_rate;
}

void
waveform_to_fbank_converter::init_computer(float32 sample_rate) const
{
    knf::MelBanksOptions mel_opts{};
    mel_opts.num_bins = opts_.num_mel_bins();

    knf::FrameExtractionOptions frame_opts{};
    frame_opts.samp_freq = sample_rate;

    knf::FbankOptions opts{};
    opts.frame_opts = frame_opts;
    opts.mel_opts = mel_opts;

    computer_ = std::make_unique<kaldi_fbank_computer>(opts);
}

}  // namespace fairseq2n
