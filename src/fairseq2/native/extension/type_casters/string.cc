// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/extension/type_casters/string.h"

#include <utility>

#include <fairseq2/native/data/immutable_string.h>

namespace pybind11::detail {

bool
type_caster<std::string>::load(pybind11::handle src, bool convert)
{
    if (isinstance<fairseq2::immutable_string>(src)) {
        value = src.cast<fairseq2::immutable_string &>().to_string();

        return true;
    } else if (subcaster_.load(src, convert)) {
        value = static_cast<std::string &&>(std::move(subcaster_));

        return true;
    }

    return false;
}

bool
type_caster<std::string_view>::load(handle src, bool convert)
{
    if (isinstance<fairseq2::immutable_string>(src)) {
        value = static_cast<std::string_view>(src.cast<fairseq2::immutable_string &>());

        return true;
    } else if (subcaster_.load(src, convert)) {
        value = static_cast<std::string_view>(subcaster_);

        return true;
    }

    return false;
}

}  // namespace pybind11::detail
