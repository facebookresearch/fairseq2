// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <optional>
#include <utility>

#include <pybind11/pybind11.h>
#include <torch/csrc/jit/python/pybind.h>

#include "fairseq2/native/data/data_source.h"
#include "fairseq2/native/intrusive_ptr.h"
#include "fairseq2/native/ivalue.h"

namespace py = pybind11;

namespace fairseq2 {
namespace {

class data_source_iterator : public intrusive_ptr_target {
public:
    explicit data_source_iterator(intrusive_ptr<data_source> &&ds) : ds_{std::move(ds)}
    {}

    data_source_iterator &
    iter() noexcept
    {
        return *this;
    }

    ivalue
    next()
    {
        bool is_eod = false;

        {
            py::gil_scoped_release no_gil{};

            is_eod = !ds_->move_next();
        }

        if (is_eod)
            throw py::stop_iteration();

        return ds_->current();
    }

private:
    intrusive_ptr<data_source> ds_;
};

intrusive_ptr<data_source>
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

    py::enum_<whence>(m,
        "Whence", "Specifies the offset origin for :py:meth:`DataSource.seek`."
    )
        .value("BEGIN",   whence::begin,   "The beginning of the data source.")
        .value("CURRENT", whence::current, "The current position.")
        .value("END",     whence::end,     "The end of the data source.");

    py::class_<data_source, intrusive_ptr<data_source>>(m,
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
        .def("__iter__", [](intrusive_ptr<data_source> ds) {
            return make_intrusive<data_source_iterator>(std::move(ds));
        },
R"docstr(
    __iter__()

    Returns an iterator that iterates over the items in the data source.

    :rtype:
        Iterator[Any]
)docstr")
        .def("reset", &data_source::reset, py::call_guard<py::gil_scoped_release>{},
R"docstr(
    reset()

    Rewinds to the beginning of the data source.
)docstr")
        .def("seek", &data_source::seek, py::arg("offset"), py::arg("whence") = whence::begin,
R"docstr(
    seek(offset, whence=Whence.FIRST)

    Seeks to a specified position in the data source.

    :param offset:
        The offset from ``whence``.
    :param whence:
        The offset origin.

    :type offset:
        int
    :type whence:
        Whence
)docstr");

    py::class_<data_source_iterator, intrusive_ptr<data_source_iterator>>(m,
        "DataSourceIterator"
    )
        .def("__iter__", &data_source_iterator::iter)
        .def("__next__", &data_source_iterator::next);

    // clang-format on
}

}  // namespace fairseq2
