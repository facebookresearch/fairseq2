// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/data_length_extractor.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <ATen/Tensor.h>

#include "fairseq2n/fmt.h"
#include "fairseq2n/data/data.h"
#include "fairseq2n/detail/exception.h"
#include "fairseq2n/utils/cast.h"

using namespace fairseq2n::detail;

namespace fairseq2n {

data_length_extractor::data_length_extractor(std::optional<std::string> maybe_selector)
{
    if (maybe_selector)
        maybe_selector_ = element_selector{*std::move(maybe_selector)};
}

std::size_t
data_length_extractor::operator()(const data &d) const
{
    auto extract_length = [this](const data &element, element_path_ref path = {})
    {
        if (element.is_tensor())
            return conditional_cast<std::size_t>(element.as_tensor().size(0));

        if (element.is_int())
            return conditional_cast<std::size_t>(element.as_int());

        if (element.is_list())
            return element.as_list().size();

        if (maybe_selector_)
            throw_<std::invalid_argument>(
                "The element at '{}' in the input data must be of type `int`, `list`, or `torch.Tensor` to determine its length, but is of type `{}` instead.", path, element.type());
        else
            throw_<std::invalid_argument>(
                "The input data must be of type `int`, `list`, or `torch.Tensor` to determine its length, but is of type `{}` instead.", element.type());
    };

    if (!maybe_selector_)
        return extract_length(d);

    std::size_t data_len = 0;

    maybe_selector_->visit(
        d, [&data_len, &extract_length](const data &element, element_path_ref path)
        {
            data_len = std::max(data_len, extract_length(element, path));
        });

    return data_len;
}

}  // namespace fairseq2n
