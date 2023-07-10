// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/data_length_extractor.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <ATen/Tensor.h>

#include "fairseq2/native/fmt.h"
#include "fairseq2/native/data/data.h"
#include "fairseq2/native/detail/exception.h"
#include "fairseq2/native/utils/cast.h"

using namespace fairseq2::detail;

namespace fairseq2 {

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

        if (selector_)
            throw_<std::invalid_argument>(
                "The element at '{}' in the input data must be of type `int`, `list`, or `torch.Tensor` to determine its length, but is of type `{}` instead.", path, element.type());
        else
            throw_<std::invalid_argument>(
                "The input data must be of type `int`, `list`, or `torch.Tensor` to determine its length, but is of type `{}` instead.", element.type());
    };

    if (!selector_)
        return extract_length(d);

    std::size_t data_length = 0;

    selector_->visit(d, [&data_length, &extract_length](const data &element, element_path_ref path)
    {
        data_length = std::max(data_length, extract_length(element, path));
    });

    return data_length;
}

}  // namespace fairseq2
