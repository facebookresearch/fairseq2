// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/flatten_data_source.h"

#include <queue>
#include <string>
#include <utility>

#include "fairseq2n/data/data.h"
#include "fairseq2n/data/detail/exception.h"

namespace fairseq2n::detail {

std::optional<data>
flatten_data_source::next()
{
    // If we have elements in the queue, return the next one
    if (!elements_queue_.empty()) {
        data element = std::move(elements_queue_.front());
        elements_queue_.pop();
        return element;
    }

    // Get the next example from the inner data source
    std::optional<data> maybe_example = inner_->next();
    if (!maybe_example)
        return std::nullopt;

    // Extract elements from the example and add them to the queue
    elements_queue_ = extract_elements(*maybe_example);

    // If the queue is empty (no elements could be extracted), recursively call next()
    if (elements_queue_.empty())
        return next();

    // Return the first element from the queue
    data element = std::move(elements_queue_.front());
    elements_queue_.pop();
    return element;
}

void
flatten_data_source::reset(bool reset_rng)
{
    // Clear the queue and reset the inner data source
    while (!elements_queue_.empty())
        elements_queue_.pop();

    inner_->reset(reset_rng);
}

void
flatten_data_source::record_position(tape &t, bool strict) const
{
    // Record the size of the queue and its contents
    t.record(elements_queue_.size());

    // Make a copy to preserve the original queue
    std::queue<data> queue_copy = elements_queue_;
    
    while (!queue_copy.empty()) {
        t.record(queue_copy.front());
        queue_copy.pop();
    }

    // Record the inner data source position
    inner_->record_position(t, strict);
}

void
flatten_data_source::reload_position(tape &t, bool strict)
{
    // Clear the current queue
    while (!elements_queue_.empty())
        elements_queue_.pop();

    // Reload the queue size
    std::size_t queue_size = t.read<std::size_t>();
    
    // Reload the queue elements
    for (std::size_t i = 0; i < queue_size; ++i)
        elements_queue_.push(t.read<data>());

    // Reload the inner data source position
    inner_->reload_position(t, strict);
}

data_source_finitude_type
flatten_data_source::finitude_type() const noexcept
{
    // The finitude type of a flattened data source is the same as its inner source
    return inner_->finitude_type();
}

std::queue<data>
flatten_data_source::extract_elements(const data &example)
{
    std::queue<data> elements;

    // If selector is provided, use it to extract elements
    if (selector_) {
        std::string error_msg = "Cannot use selector with flatten.";
        std::cerr << error_msg << std::endl;
        std::terminate();  // Force immediate program termination
    }
    // If no selector is provided, assume the example itself is a list
    else if (example.is_list()) {
        auto example_list = example.as_list();
        for (size_t i = 0; i < example_list.size(); ++i) {
            elements.push(example_list.at(i));
        }
    }
    // If no valid elements were found, the queue remains empty
    
    return elements;
}

}  // namespace fairseq2n::detail