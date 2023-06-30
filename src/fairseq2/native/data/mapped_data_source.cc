// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/mapped_data_source.h"

#include <oneapi/tbb.h>

#include "fairseq2/native/py.h"
#include "fairseq2/native/data/data_pipeline.h"

namespace fairseq2::detail {

std::optional<data>
mapped_data_source::next()
{
    if (num_parallel_calls_ <= 1) {
        std::optional<data> d{};

        while ((d = inner_->next()) && !(d = invoke_processor(*std::move(d))));

        return d;
    }

    do {
        // Yield a buffered example.
        for (; buffer_iter_ < buffer_.end(); ++buffer_iter_) {
            if (*buffer_iter_)
                return std::move(*buffer_iter_++);
        }
    // If we have exhausted all buffered examples, try to refill the buffer.
    } while (fill_buffer());

    return std::nullopt;
}

void
mapped_data_source::reset()
{
    buffer_.clear();

    buffer_iter_ = buffer_.begin();

    inner_->reset();
}

void
mapped_data_source::record_position(tape &t) const
{
    t.record(buffer_);

    t.record(buffer_iter_ - buffer_.begin());

    inner_->record_position(t);
}

void
mapped_data_source::reload_position(tape &t)
{
    buffer_ = t.read<std::vector<std::optional<data>>>();

    buffer_iter_ = buffer_.begin() + t.read<std::ptrdiff_t>();

    inner_->reload_position(t);
}

bool
mapped_data_source::fill_buffer()
{
    buffer_.clear();

    for (std::size_t i = 0; i < num_parallel_calls_; i++) {
        std::optional<data> d = inner_->next();
        if (!d)
            break;

        buffer_.push_back(std::move(d));
    }

    if (buffer_.empty())
        return false;

    // Apply the processor to all buffered examples.
    auto apply_processor = [this](const tbb::blocked_range<std::size_t> &r) {
        for (auto i = r.begin(); i < r.end(); ++i)
            buffer_[i] = invoke_processor(*std::move(buffer_[i]));
    };

    tbb::blocked_range<std::size_t> r{0, buffer_.size()};

    // Avoid threading overhead if we have just one example.
    if (buffer_.size() == 1)
        apply_processor(r);
    else
        tbb::parallel_for(r, apply_processor);

    buffer_iter_ = buffer_.begin();

    return true;
}

std::optional<data>
mapped_data_source::invoke_processor(data &&d) {
    // See the note [Python Finalization].
    if (py_is_finalizing())
        return std::nullopt;

    try {
        return processor_->process(std::move(d));
    } catch (const data_pipeline_error &) {
        if (!warn_only_)
            throw;
    } catch (...) {
        if (!warn_only_)
            data_pipeline_error::throw_nested("The map operation has failed.");
    }

    // TODO: warn

    return std::nullopt;
}

}  // namespace fairseq2::detail
