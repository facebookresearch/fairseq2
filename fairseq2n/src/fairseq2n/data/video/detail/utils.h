// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>

namespace fairseq2n::detail {

struct buffer_data {
    const uint8_t *ptr; // Pointer to the start of the memory_block buffer
    size_t size;        
};

struct media_metadata {
  int64_t num_frames{0}; // Number of frames in the stream
  int numerator{0}; // Time base numerator
  int denominator{0}; // Time base denominator
  int64_t duration_microseconds{0}; // Duration of the stream
  int height{0}; // Height of a frame in pixels
  int width{0}; // Width of a frame in pixels
  double time_base{0}; // Time base of the stream
  double fps{0}; // Frames per second for video streams
  // media_format format; // TODO
};

struct tensor_storage {
    at::Tensor all_video_frames;
    at::Tensor frame_pts;
    at::Tensor timebase;
    at::Tensor fps;
    at::Tensor duration;
};

class video_decoder_options {
public:
    video_decoder_options
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

    video_decoder_options
    maybe_device(std::optional<at::Device> value) noexcept
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

    video_decoder_options
    pin_memory(bool value) noexcept
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
    std::optional<at::ScalarType> maybe_dtype_{};
    std::optional<at::Device> maybe_device_{};
    bool pin_memory_ = false;
};

} // namespace fairseq2n::detail
