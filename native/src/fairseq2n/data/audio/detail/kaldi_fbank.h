// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <ATen/Tensor.h>
#include <kaldi-native-fbank/csrc/feature-fbank.h>
#include <kaldi-native-fbank/csrc/feature-window.h>

#include "fairseq2n/float.h"

namespace fairseq2n::detail {

class kaldi_fbank_computer {
    friend class kaldi_fbank_compute_op;

public:
    explicit
    kaldi_fbank_computer(const knf::FbankOptions &opts);

    at::Tensor
    compute(const at::Tensor &waveform, bool pin_memory);

    float32
    sample_rate() const noexcept
    {
        return opts_->frame_opts.samp_freq;
    }

private:
    knf::FbankComputer &
    native() noexcept
    {
        return native_;
    }

    const knf::FeatureWindowFunction &
    window_fn() const noexcept
    {
        return window_fn_;
    }

private:
    knf::FbankComputer native_;
    knf::FeatureWindowFunction window_fn_;
    const knf::FbankOptions *opts_;
};

}  // namespace fairseq2n::detail
