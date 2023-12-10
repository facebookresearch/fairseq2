// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/text/string_to_int_converter.h"

#include <stdexcept>

#include "fairseq2n/fmt.h"
#include "fairseq2n/utils/string.h"
#include "fairseq2n/data/immutable_string.h"
#include "fairseq2n/detail/exception.h"

using namespace fairseq2n::detail;

namespace fairseq2n {

data
string_to_int_converter::operator()(data &&d) const
{
    if (!d.is_string())
        throw_<std::invalid_argument>(
            "The input data must be of type `string`, but is of type `{}` instead.", d.type());

    const immutable_string &s = d.as_string();

    try {
        return from_string<std::int64_t>(s, base_);
    } catch (const std::out_of_range &) {
        throw_<std::invalid_argument>(
            "The input string must represent a 64-bit integer, but is '{}' instead, which is out of range.", s);
    } catch (const std::invalid_argument &) {
        throw_<std::invalid_argument>(
            "The input string must represent a 64-bit integer, but is '{}' instead.", s);
    }
}

}  // namespace fairseq2n
