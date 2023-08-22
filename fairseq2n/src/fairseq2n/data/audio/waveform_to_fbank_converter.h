// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

#include <ATen/Device.h>
#include <ATen/ScalarType.h>

#include "fairseq2n/api.h"
#include "fairseq2n/float.h"
#include "fairseq2n/data/data.h"

namespace fairseq2n {

class fbank_options {
public:
    fbank_options
    num_mel_bins(std::int32_t value) noexcept
    {
        auto tmp = *this;

        tmp.num_mel_bins_ = value;

        return tmp;
    }

    std::int32_t
    num_mel_bins() const noexcept
    {
        return num_mel_bins_;
    }

    fbank_options
    waveform_scale(float32 value) noexcept
    {
        auto tmp = *this;

        tmp.waveform_scale_ = value;

        return tmp;
    }

    float32
    waveform_scale() const noexcept
    {
        return waveform_scale_;
    }

    fbank_options
    channel_last(bool value) noexcept
    {
        auto tmp = *this;

        tmp.channel_last_ = value;

        return tmp;
    }

    bool
    channel_last() const noexcept
    {
        return channel_last_;
    }

    fbank_options
    standardize(bool value) noexcept
    {
        auto tmp = *this;

        tmp.standardize_ = value;

        return tmp;
    }

    bool
    standardize() const noexcept
    {
        return standardize_;
    }

    fbank_options
    keep_waveform(bool value) noexcept
    {
        auto tmp = *this;

        tmp.keep_waveform_ = value;

        return tmp;
    }

    bool
    keep_waveform() const noexcept
    {
        return keep_waveform_;
    }

    fbank_options
    maybe_dtype(std::optional<at::ScalarType> value) noexcept
    {
        auto tmp = *this;

        tmp.maybe_dtype_ = value;

        return tmp;
    }

    std::optional<at::ScalarType>
    maybe_dtype() const noexcept
    {
        return maybe_dtype_;
    }

    fbank_options
    maybe_device(std::optional<at::Device> value) && noexcept
    {
        auto tmp = *this;

        tmp.maybe_device_ = value;

        return tmp;
    }

    std::optional<at::Device>
    maybe_device() const noexcept
    {
        return maybe_device_;
    }

    fbank_options
    pin_memory(bool value) && noexcept
    {
        auto tmp = *this;

        tmp.pin_memory_ = value;

        return tmp;
    }

    bool
    pin_memory() const noexcept
    {
        return pin_memory_;
    }

private:
    std::int32_t num_mel_bins_ = 80;
    float32 waveform_scale_ = 1.0;
    bool channel_last_ = false;
    bool standardize_ = false;
    bool keep_waveform_ = false;
    std::optional<at::ScalarType> maybe_dtype_{};
    std::optional<at::Device> maybe_device_{};
    bool pin_memory_ = false;
};

namespace detail {

class kaldi_fbank_computer;

}

class FAIRSEQ2_API waveform_to_fbank_converter {
public:
    explicit
    waveform_to_fbank_converter(fbank_options opts = {}) noexcept;

    waveform_to_fbank_converter(const waveform_to_fbank_converter &) = delete;
    waveform_to_fbank_converter &operator=(const waveform_to_fbank_converter &) = delete;

    waveform_to_fbank_converter(waveform_to_fbank_converter &&other) = delete;
    waveform_to_fbank_converter &operator=(waveform_to_fbank_converter &&other) = delete;

   ~waveform_to_fbank_converter();

    data
    operator()(data &&d) const;

private:
    static at::Tensor &
    find_waveform(data_dict &dict);

    static float32
    find_sample_rate(const data_dict &dict);

    void
    init_computer(float32 sample_rate) const;

private:
    mutable std::unique_ptr<detail::kaldi_fbank_computer> computer_;
    mutable std::mutex init_mutex_{};
    fbank_options opts_;
};

}  // namespace fairseq2n
