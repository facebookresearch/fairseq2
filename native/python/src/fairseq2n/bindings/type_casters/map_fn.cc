// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/bindings/type_casters/map_fn.h"

#include <fairseq2n/data/py.h>
#include <fairseq2n/data/text/string_to_int_converter.h>
#include <fairseq2n/data/text/string_to_tensor_converter.h>

#include "fairseq2n/bindings/type_casters/data.h"
#include "fairseq2n/bindings/type_casters/string.h"

namespace py = pybind11;

namespace fairseq2n {

map_fn
map_functor_registry::maybe_as_functor(py::handle src)
{
    if (auto pos = types_.find(py::type::of(src)); pos != types_.end())
        return pos->second(src);

    return nullptr;
}

void
map_functor_registry::register_(py::type &&t, cast_to_map_fn &&fn)
{
    types_[std::move(t)] = std::move(fn);
}

map_functor_registry &
map_functors() noexcept
{
    static map_functor_registry registry{};

    return registry;
}

}  // namespace fairseq2n

using namespace fairseq2n;

namespace pybind11::detail {

bool
type_caster<map_fn>::load(handle src, bool)
{
    if (map_fn fn = map_functors().maybe_as_functor(src); fn != nullptr) {
        value = std::move(fn);

        return true;
    }

    static module_ builtins = module_::import("builtins");

    // int
    if (src.is(builtins.attr("int"))) {
        value = [](data &&d)
        {
            return string_to_int_converter{}(std::move(d));
        };

        return true;
    }

    static module_ torch = module_::import("torch");

    // Tensor
    if (src.is(torch.attr("tensor"))) {
        value = [](data &&d)
        {
            return string_to_tensor_converter{}(std::move(d));
        };

        return true;
    }

    // Callable
    if (isinstance<function>(src)) {
        class move_fn_wrapper {
        public:
            explicit
            move_fn_wrapper(function &&fn) noexcept
              : fn_{std::move(fn)}
            {}

            move_fn_wrapper(const move_fn_wrapper &) = default;
            move_fn_wrapper &operator=(const move_fn_wrapper &) = default;

            move_fn_wrapper(move_fn_wrapper &&) = default;
            move_fn_wrapper &operator=(move_fn_wrapper &&) = default;

           ~move_fn_wrapper()  // NOLINT(bugprone-exception-escape)
            {
                gil_scoped_acquire gil{};

                fn_ = {};
            }

            data
            operator()(data &&d)
            {
                gil_scoped_acquire gil{};

                return fn_(std::move(d)).cast<data>();
            }

        private:
            function fn_;
        };

        value = move_fn_wrapper{src.cast<function>()};

        return true;
    }

    return false;
}

}  // namespace pybind11::detail
