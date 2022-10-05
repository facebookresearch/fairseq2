// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <optional>
#include <utility>

#include <pybind11/pybind11.h>
#include <torch/csrc/jit/python/pybind.h>

#include <fairseq2/native/data/data_source.h>

namespace py = pybind11;

using fairseq2::data_source;

namespace {

class data_source_iterator : public c10::intrusive_ptr_target {
public:
    explicit data_source_iterator(c10::intrusive_ptr<data_source> &&ds) : ds_{std::move(ds)}
    {}

    data_source_iterator &
    iter() noexcept
    {
        return *this;
    }

    c10::IValue
    next()
    {
        bool is_eod = false;

        {
            py::gil_scoped_release no_gil{};

            is_eod = !ds_->move_next();
        }

        if (is_eod) {
            throw py::stop_iteration();
        }

        return ds_->current();
    }

private:
    c10::intrusive_ptr<data_source> ds_;
};

c10::intrusive_ptr<data_source>
list_files(const std::vector<std::string> &paths, const std::optional<std::string> &pattern)
{
    return data_source::list_files(paths, pattern);
}

}  // namespace

PYBIND11_MODULE(data_source, m)
{
    py::options opts{};
    opts.disable_function_signatures();

    // clang-format off

    py::class_<data_source, c10::intrusive_ptr<data_source>>(m,
        "DataSource",
R"docstr(
    Lorem ipsum
)docstr")
        .def_static(
            "list_files",
            &list_files,
            py::arg("paths"),
            py::arg("pattern").none(true),
            py::call_guard<py::gil_scoped_release>{},
R"docstr(
    list_files(paths, pattern=None)

    Lorem ipsum

    :param paths:
        List of files.
    :param pattern:
        The pattern.

    :type paths:
        List[str]
    :type pattern:
        Optional[str]

    :rtype:
        DataSource
)docstr")
        .def("__iter__", [](c10::intrusive_ptr<data_source> ds) {
            return c10::make_intrusive<data_source_iterator>(std::move(ds));
        });

    py::class_<data_source_iterator, c10::intrusive_ptr<data_source_iterator>>(m,
        "DataSourceIterator"
    )
        .def("__iter__", &data_source_iterator::iter)
        .def("__next__", &data_source_iterator::next);

    // clang-format on
}
