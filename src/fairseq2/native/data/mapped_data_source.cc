// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/mapped_data_source.h"

#include <stdexcept>

namespace fairseq2::detail {

std::optional<data>
mapped_data_source::next()
{
    std::optional<data> d = inner_->next();
    if (!d)
        return {};

    try {
        return fn_(*std::move(d));
    } catch (const data_pipeline_error &) {
        throw;
    } catch (const std::invalid_argument &) {
        data_pipeline_error::throw_nested("The map function has failed.", std::move(d));
    } catch (...) {
        data_pipeline_error::throw_nested("The map function has failed.");
    }
}

std::size_t
mapped_data_source::skip(std::size_t num_examples)
{
    return inner_->skip(num_examples);
}

void
mapped_data_source::reset()
{
    inner_->reset();
}

void
mapped_data_source::record_position(tape &t) const
{
    inner_->record_position(t);
}

void
mapped_data_source::reload_position(tape &t)
{
    inner_->reload_position(t);
}

}  // namespace fairseq2::detail
