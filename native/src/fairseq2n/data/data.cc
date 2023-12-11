// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/data.h"

#include <stdexcept>

#include "fairseq2n/detail/exception.h"

using namespace fairseq2n::detail;

namespace fairseq2n {

std::string
repr<data_type>::operator()(data_type dt) const
{
    switch (dt) {
    case data_type::bool_:
        return "bool";
    case data_type::int_:
        return "int";
    case data_type::float_:
        return "float";
    case data_type::string:
        return "string";
    case data_type::tensor:
        return "torch.Tensor";
    case data_type::memory_block:
        return "memory_block";
    case data_type::list:
        return "list";
    case data_type::dict:
        return "dict";
    case data_type::pyobj:
        return "pyobj";
    };

    throw_<std::invalid_argument>("`dt` is not a valid data type.");
}

}  // namespace fairseq2n
