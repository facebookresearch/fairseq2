// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/element_mapper.h"

#include <exception>
#include <stdexcept>

#include "fairseq2n/fmt.h"
#include "fairseq2n/detail/exception.h"

using namespace fairseq2n::detail;

namespace fairseq2n {

element_mapper::element_mapper(map_fn fn, std::optional<std::string> maybe_selector)
  : map_fn_{std::move(fn)}
{
    if (maybe_selector)
        maybe_selector_ = element_selector{*std::move(maybe_selector)};
}
data
element_mapper::operator()(data &&d)
{
    if (!maybe_selector_)
        return map_fn_(std::move(d));

    maybe_selector_->visit(d, [this](data &element, element_path_ref path)
    {
        try {
            element = map_fn_(std::move(element));
        } catch (const std::exception &) {
            throw_with_nested<std::runtime_error>(
                "The map function has failed while processing the path '{}' of the input data. See nested exception for details.", path);
        }
    });

    return std::move(d);
}

}  // namespace fairseq2n
