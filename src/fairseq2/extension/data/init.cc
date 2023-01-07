// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/extension/data/init.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <iterator>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/core.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <fairseq2/native/data/data.h>
#include <fairseq2/native/data/data_pipeline.h>
#include <fairseq2/native/data/data_processor.h>
#include <fairseq2/native/data/immutable_string.h>
#include <fairseq2/native/data/record_reader.h>
#include <fairseq2/native/data/stream.h>
#include <fairseq2/native/data/tape.h>
#include <fairseq2/native/utils/string.h>

#include "fairseq2/extension/data/text/init.h"
#include "fairseq2/extension/type_casters/type_casters.h"

namespace py = pybind11;

using namespace fairseq2::detail;

namespace fairseq2 {
namespace {

class data_pipeline_iterator {
public:
    explicit
    data_pipeline_iterator(data_pipeline &dp) noexcept
        : data_pipeline_{&dp}
    {}

    data
    next()
    {
        std::optional<data> d = data_pipeline_->next();
        if (!d)
            throw py::stop_iteration();

        return *std::move(d);
    }

private:
    data_pipeline *data_pipeline_;
};

void
def_data_pipeline(py::module_ &base)
{
    py::module_ m = base.def_submodule("data_pipeline");

    py::class_<data_pipeline>(m, "DataPipeline",
        R"(
        Lorem ipsum.
        )")
        .def(py::init<>())
        .def("__iter__",
            [](data_pipeline &self)
            {
                return data_pipeline_iterator{self};
            },
            py::keep_alive<0, 1>{})
        .def("skip", &data_pipeline::skip, py::arg("num_examples"),
            R"(
            Skips reading a specified number of examples.

            Returns:
                The number of examples skipped. It can be less than num_examples
                if the end of the data pipeline is reached.
            )")
        .def("reset", &data_pipeline::reset,
            R"(
            Moves back to the first example.
            )")
        .def("record_position", &data_pipeline::record_position, py::arg("t"),
            R"(
            Records the current position of the data pipeline to t.
            )")
        .def("reload_position", &data_pipeline::reload_position, py::arg("t"),
            R"(
            Reloads the current position of the data pipeline from t.
            )")
        .def_property_readonly("is_broken", &data_pipeline::is_broken,
            R"(
            Indicates whether the data pipeline is broken by a previous operation.
            )");

    py::class_<data_pipeline_iterator>(m, "DataPipelineIterator")
        .def("__iter__",
            [](data_pipeline_iterator &self) -> data_pipeline_iterator &
            {
                return self;
            })
        .def("__next__", &data_pipeline_iterator::next);

    py::class_<data_pipeline_builder>(m, "DataPipelineBuilder")
        .def("batch",
            [](data_pipeline_builder &self, std::size_t batch_size, bool drop_remainder) -> decltype(auto)
            {
                return self.batch(batch_size, drop_remainder);
            },
            py::arg("batch_size"),
            py::arg("drop_remainder"),
            R"(
            Combines a number of consecutive examples into a single example.
            )")
        .def("map",
            [](data_pipeline_builder &self, const data_processor &dp) -> decltype(auto)
            {
                auto fn = [nurse = py::cast(dp).cast<py_object>(), &dp](data &&d) {
                    return dp(std::move(d));
                };

                return self.map(std::move(fn));
            },
            py::arg("dp"))
        .def("map",
            [](data_pipeline_builder &self, map_fn &&fn) -> decltype(auto)
            {
                return self.map(std::move(fn));
            },
            py::arg("fn"),
            R"(
            Applies fn to every example in the data pipeline.
            )")
        .def("shard",
            [](data_pipeline_builder &self, std::size_t shard_idx, std::size_t num_shards) -> decltype(auto)
            {
                return self.shard(shard_idx, num_shards);
            },
            py::arg("shard_idx"),
            py::arg("num_shards"),
            R"(
            Reads only 1/num_shards of all examples in the data pipeline.
            )")
        .def("yield_from",
            [](data_pipeline_builder &self, yield_fn &&fn) -> decltype(auto)
            {
                return self.yield_from(std::move(fn));
            },
            py::arg("fn"),
            R"(
            Applies fn to every example in the data pipeline and yields the
            examples from the returned sub-data pipelines.
            )")
        .def("and_return",
            [](data_pipeline_builder &self)
            {
                return std::move(self).and_return();
            },
            R"(
            Returns a new DataPipeline instance.
            )");

    py::class_<data_processor>(m, "DataProcessor")
        .def("__call__", &data_processor::operator());

    static py::exception<data_pipeline_error> py_data_pipeline_error{
        m, "DataPipelineError", PyExc_RuntimeError};

    m.def("list_files", &list_files, py::arg("pathname"), py::arg("pattern") = "");

    m.def("read_sequence", &read_list, py::arg("s"),
        R"(
        Returns a data pipeline from s.
        )");

    m.def("zip_data_pipelines",
        [](std::vector<std::reference_wrapper<data_pipeline>> &zip)
        {
            std::vector<data_pipeline> c{};

            c.reserve(zip.size());

            std::transform(zip.begin(), zip.end(), std::back_inserter(c), [](auto &i) {
                return std::move(i.get());
            });

            return zip_data_pipelines(std::move(c));
        },
        py::arg("data_pipelines"),
        R"(
        Builds a data pipeline by zipping together data_pipelines.
        )");

    static py::exception<stream_error> py_stream_error{m, "StreamError", PyExc_RuntimeError};

    static py::exception<record_error> py_record_error{m, "RecordError", PyExc_RuntimeError};

    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    py::register_exception_translator([](std::exception_ptr ptr)
    {
        if (!ptr)
            return;

        auto raise_error = [&ptr](const std::exception &e, const py::object &err) {
            py::detail::raise_err(err.ptr(), e.what());

            py::detail::handle_nested_exception(e, ptr);
        };

        try {
            std::rethrow_exception(ptr);
        } catch (const stream_error &e) {
            raise_error(e, py_stream_error);
        } catch (const record_error &e) {
            raise_error(e, py_record_error);
        } catch (const data_pipeline_error &e) {
            raise_error(e, py_data_pipeline_error);
        }
    });
}

void
def_string(py::module_ &base)
{
    py::module_ m = base.def_submodule("string");

    py::class_<immutable_string>(m, "String",
        R"(
        Represents an immutable UTF-8 string that supports zero-copy marshalling
        between Python and native code.
        )")
        .def(py::init<>())
        .def(py::init<std::string_view>(), py::arg("s"))

        // To be consistent with str, we return the UTF-8 code point length
        // instead of the byte length.
        .def("__len__", &immutable_string::get_code_point_length)

        .def(py::self == py::self)  // NOLINT(misc-redundant-expression)
        .def(py::self != py::self)  // NOLINT(misc-redundant-expression)

        // Equality check with other string-likes.
        .def("__eq__",
            [](const immutable_string &lhs, std::string_view rhs)
            {
                return static_cast<std::string_view>(lhs) == rhs;
            })
        .def("__ne__",
            [](const immutable_string &lhs, std::string_view rhs)
            {
                return static_cast<std::string_view>(lhs) != rhs;
            })

        .def(py::hash(py::self))

        .def("__str__",
            [](const immutable_string &self)
            {
                return static_cast<std::string_view>(self);
            })
        .def("__repr__",
            [](const immutable_string &self)
            {
                return fmt::format("String('{}')", self);
            })

        .def("lstrip",
            [](const immutable_string &self)
            {
                return ltrim(self);
            },
            R"(
            Returns a copy of the string with no whitespace at the beginning.
            )")
        .def("rstrip",
            [](const immutable_string &self)
            {
                return rtrim(self);
            },
            R"(
            Returns a copy of the string with no whitespace at the end.
            )")
        .def("to_py", &immutable_string::operator std::string_view,
            R"(
            Converts to str.
            )")

        .def(py::pickle(
            [](const immutable_string &self)
            {
                return py::cast(static_cast<std::string_view>(self));
            },
            [](const py::object &o) -> immutable_string
            {
                return o.cast<std::string_view>();
            }));

    py::implicitly_convertible<std::string_view, immutable_string>();
}

void
def_tape(py::module_ &base)
{
    py::module_ m = base.def_submodule("tape");

    py::class_<tape>(m, "Tape",
        R"(
        Lorem ipsum.
        )")
        .def(py::init<>())

        .def("rewind", &tape::rewind,
            R"(
            Rewinds back to the beginning of the tape.
            )")

        .def(py::pickle(
            [](const tape &self)
            {
                return tape_attorney::get_storage(self);
            },
            [](std::vector<data> &&storage)
            {
                return tape_attorney::make(std::move(storage));
            }));
}

}  // namespace

void
def_data(py::module_ &base)
{
    py::module_ m = base.def_submodule("data");

    def_data_pipeline(m);

    def_string(m);

    def_tape(m);

    def_text(m);
}

}  // namespace fairseq2
