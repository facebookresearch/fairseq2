// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/bindings/type_casters/string.h"

#include <optional>
#include <utility>

#include <fairseq2n/data/immutable_string.h>

using namespace fairseq2n;

namespace pybind11::detail {
namespace {

object
maybe_as_pathname(handle src) {
    // Check if we have a path-like object.
    object fspath = getattr(src, "__fspath__", none());

    if (fspath.is_none())
        return fspath;
    else
        // Return the string representation of the path.
        return fspath();
}

}  // namespace

bool
type_caster<std::string>::load(handle src, bool convert)
{
    if (isinstance<immutable_string>(src)) {
        value = src.cast<immutable_string &>().to_string();

        return true;
    }

    if (inner_caster_.load(src, convert)) {
        value = static_cast<std::string &&>(std::move(inner_caster_));

        return true;
    }

    object pathname = maybe_as_pathname(src);

    if (inner_caster_.load(pathname, convert)) {
        value = static_cast<std::string &&>(std::move(inner_caster_));

        return true;
    }

    return false;
}

bool
type_caster<std::string_view>::load(handle src, bool convert)
{
    if (isinstance<immutable_string>(src)) {
        value = static_cast<std::string_view>(src.cast<immutable_string &>());

        return true;
    }

    if (inner_caster_.load(src, convert)) {
        value = static_cast<std::string_view>(inner_caster_);

        return true;
    }

    object pathname = maybe_as_pathname(src);

    if (inner_caster_.load(pathname, convert)) {
        value = static_cast<std::string_view>(inner_caster_);

        // We have to keep the pathname alive until the enclosing function
        // returns.
        loader_life_support::add_patient(pathname);

        return true;
    }

    return false;
}

}  // namespace pybind11::detail
