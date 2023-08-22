// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/audio/detail/kaldi_fbank.h"

#include <cstdint>
#include <vector>
#include <utility>

#include <ATen/Functions.h>
#include <ATen/Storage.h>
#include <oneapi/tbb.h>

#include "fairseq2n/memory.h"
#include "fairseq2n/span.h"

namespace fairseq2n::detail {

class kaldi_fbank_compute_op {
public:
    explicit
    kaldi_fbank_compute_op(
        kaldi_fbank_computer *computer, const at::Tensor &waveform, bool pin_memory) noexcept
      : computer_{computer}, waveform_{&waveform}, pin_memory_{pin_memory}
    {
        num_mel_bins_ = computer_->native().Dim();
    }

    at::Tensor &&
    run() &&;

private:
    span<float32>
    get_fbank_storage() noexcept;

    span<const float32>
    get_waveform_storage() const noexcept;

private:
    kaldi_fbank_computer *computer_;
    const at::Tensor *waveform_;
    bool pin_memory_;
    std::int32_t num_mel_bins_;
    at::Tensor fbank_{};
};

at::Tensor &&
kaldi_fbank_compute_op::run() &&
{
    const knf::FrameExtractionOptions &frame_opts = computer_->native().GetFrameOptions();

    // Compute the number of frames that the fbank tensor will have based on the
    // number of waveform samples.
    std::int32_t num_frames = knf::NumFrames(/*num_samples=*/waveform_->size(1), frame_opts);

    fbank_ = at::empty({num_frames, num_mel_bins_},
        at::dtype(at::kFloat).device(at::kCPU).pinned_memory(pin_memory_));

    span<float32> fbank_data = get_fbank_storage();

    span<const float32> waveform_data = get_waveform_storage();

    auto compute_fbank = [this, &frame_opts, &fbank_data, &waveform_data](
        const tbb::blocked_range<std::int32_t> &range)
    {
        std::vector<float32> signal_frame{};

        for (std::int32_t frame_nr = range.begin(); frame_nr < range.end(); ++frame_nr) {
            signal_frame.resize(0);

            // Extract the frame from the waveform tensor.
            knf::ExtractWindow(
                /*sample_offset=*/0,
                waveform_data.data(),
                waveform_data.size(),
                frame_nr,
                frame_opts,
                computer_->window_fn(),
                &signal_frame);

            span<float32> output = fbank_data.subspan(
                static_cast<std::size_t>(num_mel_bins_) * static_cast<std::size_t>(frame_nr),
                static_cast<std::size_t>(num_mel_bins_));

            // And, write it to the fbank tensor.
            computer_->native().Compute(
                /*signal_raw_log_energy=*/0, /*vtln_warp=*/1.0, &signal_frame, output.data());
        }
    };

    tbb::blocked_range<std::int32_t> range{0, num_frames};

    tbb::parallel_for(range, compute_fbank);

    return std::move(fbank_);
}

span<float32>
kaldi_fbank_compute_op::get_fbank_storage() noexcept
{
    const at::Storage &storage = fbank_.storage();

    writable_memory_span bits{storage.unsafe_data<std::byte>(), storage.nbytes()};

    return cast<float32>(bits);
}

span<const float32>
kaldi_fbank_compute_op::get_waveform_storage() const noexcept
{
    const at::Storage &storage = waveform_->storage();

    memory_span bits{storage.unsafe_data<std::byte>(), storage.nbytes()};

    return cast<const float32>(bits);
}

kaldi_fbank_computer::kaldi_fbank_computer(const knf::FbankOptions &opts)
  : native_{opts}, window_fn_{native_.GetFrameOptions()}, opts_{&native_.GetOptions()}
{}

at::Tensor
kaldi_fbank_computer::compute(const at::Tensor &waveform, bool pin_memory)
{
    return kaldi_fbank_compute_op{this, waveform, pin_memory}.run();
}

}  // namespace fairseq2n::detail
