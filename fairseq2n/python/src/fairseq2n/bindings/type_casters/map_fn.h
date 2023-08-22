// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>

#include <pybind11/pybind11.h>

#include <fairseq2n/data/data_pipeline.h>

namespace fairseq2n {

// We use this registry to support C++ functors that can be directly called from
// the Python API without acquiring GIL. A registered functor must be compatible
// with the `map_fn` signature.
class map_functor_registry {
    using cast_to_map_fn = std::function<map_fn(pybind11::handle)>;

    struct py_handle_hash {
        std::size_t
        operator()(const pybind11::handle &h) const noexcept
        {
            return std::hash<void *>{}(h.ptr());
        }
    };

public:
    // Checks if the C++ type of `src` is a registered functor. If yes, returns
    // `src` as a native callable 
    map_fn
    maybe_as_functor(pybind11::handle src);

    // Registers `T` as a functor.
    template <typename T>
    void
    register_();

private:
    void
    register_(pybind11::type &&t, cast_to_map_fn &&fn);

private:
    std::unordered_map<pybind11::type, cast_to_map_fn, py_handle_hash> types_{};
};

template <typename T>
void
map_functor_registry::register_()
{
    auto fn = [](pybind11::handle src)
    {
        // We require functor wrappers to use `std::shared_ptr` as the holder
        // type; otherwise, we cannot reliably share them.
        return [functor = src.cast<std::shared_ptr<const T>>()](data &&d)
        {
            return (*functor)(std::move(d));
        };
    };

    register_(pybind11::type::of<T>(), std::move(fn));
}

// The singleton registry instance.
map_functor_registry &
map_functors() noexcept;

}  // namespace fairseq2n

namespace pybind11::detail {

// We have special handling for callables passed to the `map()` data pipeline
// operator. Our C++ API offers various functors that can be passed to `map()`
// via their Python bindings. For such functors, we want to avoid the cost of
// going through Python. This `type_caster` supports such native functors and
// can call them directly in C++.
template <>
struct type_caster<fairseq2n::map_fn> {
    PYBIND11_TYPE_CASTER(fairseq2n::map_fn, const_name("Callable[[Any], Any]"));

public:
    bool
    load(handle src, bool);

    static handle
    cast(const fairseq2n::map_fn &fn, return_value_policy policy, handle)
    {
        return cpp_function{fn, policy}.release();
    }

    static handle
    cast(fairseq2n::map_fn &&fn, return_value_policy policy, handle)
    {
        return cpp_function{std::move(fn), policy}.release();
    }
};

}  // namespace pybind11::detail
