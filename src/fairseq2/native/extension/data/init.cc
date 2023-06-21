// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/extension/module.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>
#include <variant>

#include <fmt/core.h>

#include <fairseq2/native/data/zipfile_data_source.h>
#include <fairseq2/native/data/data.h>
#include <fairseq2/native/data/data_pipeline.h>
#include <fairseq2/native/data/data_processor.h>
#include <fairseq2/native/data/immutable_string.h>
#include <fairseq2/native/data/record_reader.h>
#include <fairseq2/native/data/stream.h>
#include <fairseq2/native/data/tape.h>
#include <fairseq2/native/utils/cast.h>
#include <fairseq2/native/utils/string.h>

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

    py::class_<data_pipeline>(m, "DataPipeline")
        .def(py::init<>())

        .def("__iter__",
            [](data_pipeline &self)
            {
                return data_pipeline_iterator{self};
            },
            py::keep_alive<0, 1>{})

        .def("reset", &data_pipeline::reset)

        .def_property_readonly("is_broken", &data_pipeline::is_broken)

        .def("state_dict",
            [](const data_pipeline &self)
            {
                tape t{};

                self.record_position(t);

                return py::dict{py::arg("position") = py::cast(t.storage())};
            })
        .def("load_state_dict",
            [](data_pipeline &self, const py::dict &state_dict, bool strict)
            {
                py::object value;
                try {
                    value = state_dict["position"];
                } catch (const py::error_already_set &ex) {
                    if (ex.matches(PyExc_KeyError) && !strict)
                        return;

                    throw;
                }

                std::vector<data> storage;
                try {
                    storage = value.cast<std::vector<data>>();
                } catch (const py::cast_error &) {
                    throw std::invalid_argument{
                        "The specified data pipeline state is corrupt."};
                }

                tape t{std::move(storage)};

                self.reload_position(t);
            },
            py::arg("state_dict"),
            py::arg("strict") = true)
        .def_static("zip",
            [](std::vector<std::reference_wrapper<data_pipeline>> &refs, bool warn_only, bool disable_parallelism)
            {
                std::vector<data_pipeline> pipelines{};

                pipelines.reserve(refs.size());

                std::transform(refs.begin(), refs.end(), std::back_inserter(pipelines), [](auto &r) {
                    return std::move(r.get());
                });

                return data_pipeline::zip(std::move(pipelines), warn_only, disable_parallelism);
            },
            py::arg("pipelines"),
            py::arg("warn_only") = false,
            py::arg("disable_parallelism") = false)
        .def_static("round_robin",
            [](std::vector<std::reference_wrapper<data_pipeline>> &refs)
            {
                std::vector<data_pipeline> pipelines{};

                pipelines.reserve(refs.size());

                std::transform(refs.begin(), refs.end(), std::back_inserter(pipelines), [](auto &r) {
                    return std::move(r.get());
                });

                return data_pipeline::round_robin(std::move(pipelines));
            },
            py::arg("pipelines"));


    py::class_<data_pipeline_iterator>(m, "_DataPipelineIterator")
        .def("__iter__",
            [](data_pipeline_iterator &self) -> data_pipeline_iterator &
            {
                return self;
            })
        .def("__next__", &data_pipeline_iterator::next);

    py::class_<data_pipeline_builder>(m, "DataPipelineBuilder")
        .def("batch",
            [](data_pipeline_builder &self, std::size_t batch_size, bool drop_remainder,
             const std::variant<std::vector<std::int32_t>, std::int32_t> &pad_idx)
                -> data_pipeline_builder &
            {
                std::vector<std::int32_t> padding{};
                if (std::holds_alternative<std::int32_t>(pad_idx))
                    padding = {std::get<std::int32_t>(pad_idx)};
                else
                    padding = std::get<std::vector<std::int32_t>>(pad_idx);

                self = std::move(self).batch(batch_size, drop_remainder, padding);

                return self;
            },
            py::arg("batch_size"),
            py::kw_only(),
            py::arg("drop_remainder") = false,
            py::arg("pad_idx") = std::vector<std::int32_t>{})
        .def("batch_by_length",
            [](
                data_pipeline_builder &self,
                std::vector<std::pair<std::size_t, std::size_t>> &buffer_sizes,
                std::int32_t pad_idx
            ) -> data_pipeline_builder &
            {
                self = std::move(self).batch_by_length(buffer_sizes, pad_idx);

                return self;
            },
            py::arg("buffer_sizes"),
            py::arg("pad_idx"))
        .def("filter",
            [](data_pipeline_builder &self, predicate_fn &&fn) -> data_pipeline_builder &
            {
                self = std::move(self).filter(std::move(fn));

                return self;
            },
            py::arg("fn"))
        .def("map",
            [](data_pipeline_builder &self, const data_processor &dp, std::size_t num_parallel_calls)
                -> data_pipeline_builder &
            {
                auto fn = [nurse = py::cast(dp).cast<py_object>(), &dp](data &&d) {
                    return dp(std::move(d));
                };

                self = std::move(self).map(std::move(fn), num_parallel_calls);

                return self;
            },
            py::arg("dp"),
            py::arg("num_parallel_calls") = 1)
        .def("map",
            [](data_pipeline_builder &self, map_fn &&fn, std::size_t num_parallel_calls)
                -> data_pipeline_builder &
            {
                self = std::move(self).map(std::move(fn), num_parallel_calls);

                return self;
            },
            py::arg("fn"),
            py::arg("num_parallel_calls") = 1)
        .def("prefetch",
            [](data_pipeline_builder &self, std::size_t num_examples) -> data_pipeline_builder &
            {
                self = std::move(self).prefetch(num_examples);

                return self;
            },
            py::arg("num_examples"))
        .def("shard",
            [](data_pipeline_builder &self, std::size_t shard_idx, std::size_t num_shards) -> data_pipeline_builder &
            {
                if (num_shards <= 0) {
                    throw std::invalid_argument("shard: num_shards must be > 0");
                }
                if (shard_idx >= num_shards) {
                    throw std::invalid_argument("shard: shard_idx must be between 0 and num_shards");
                }

                self = std::move(self).shard(shard_idx, num_shards);

                return self;
            },
            py::arg("shard_idx"),
            py::arg("num_shards"))
        .def("shuffle",
            [](data_pipeline_builder &self, std::size_t shuffle_window, bool strict) -> data_pipeline_builder &
            {
                self = std::move(self).shuffle(shuffle_window, strict);

                return self;
            },
            py::arg("shuffle_window"),
            py::arg("strict") = true)
        .def("skip",
            [](data_pipeline_builder &self, std::size_t num_examples) -> data_pipeline_builder &
            {
                self = std::move(self).skip(num_examples);

                return self;
            },
            py::arg("num_examples"))
        .def("take",
            [](data_pipeline_builder &self, std::size_t num_examples) -> data_pipeline_builder &
            {
                self = std::move(self).take(num_examples);

                return self;
            },
            py::arg("num_examples"))
        .def("yield_from",
            [](data_pipeline_builder &self, yield_fn &&fn) -> data_pipeline_builder &
            {
                self = std::move(self).yield_from(std::move(fn));

                return self;
            },
            py::arg("fn"))
        .def("and_return",
            [](data_pipeline_builder &self) -> data_pipeline
            {
                return std::move(self).and_return();
            });

    py::class_<data_processor>(m, "_DataProcessor")
        .def("__call__", &data_processor::operator(), py::call_guard<py::gil_scoped_release>{});

    static py::exception<data_pipeline_error> py_data_pipeline_error{
        m, "DataPipelineError", PyExc_RuntimeError};

    m.def("list_files", &list_files, py::arg("pathname"), py::arg("pattern") = std::nullopt);

    m.def("read_sequence", &read_list, py::arg("seq"));

    m.def("read_zipped_records", &read_zipped_records, py::arg("pathname"));

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

std::size_t
compute_py_buffer_size(const py::buffer_info &info)
{
    py::ssize_t size = info.itemsize;

    for (std::ptrdiff_t i = ssize(info.shape) - 1; i >= 0; i--) {
        auto dim = static_cast<std::size_t>(i);

        if (info.strides[dim] == size)
            size *= info.shape[dim];
        else
            throw std::invalid_argument{"The specified buffer must be contiguous."};
    }

    return static_cast<std::size_t>(size);
}

void
release_py_buffer(const void *, std::size_t, void *ctx) noexcept  // NOLINT(bugprone-exception-escape)
{
    py::gil_scoped_acquire gil{};

    PyBuffer_Release(static_cast<Py_buffer *>(ctx));
}

void
def_memory(py::module_ &base)
{
    py::module_ m = base.def_submodule("memory");

    py::class_<memory_block>(m, "MemoryBlock", py::buffer_protocol())
        .def(py::init<>())
        .def(py::init(
            [](const py::buffer &b, bool copy) -> memory_block
            {
                py::buffer_info info = b.request();

                auto data = static_cast<memory_block::const_pointer>(info.ptr);

                std::size_t size = compute_py_buffer_size(info);

                if (copy)
                    return copy_memory({data, size});

                Py_buffer *buf = std::exchange(info.view(), nullptr);

                return memory_block{data, size, buf, release_py_buffer};
            }),
            py::arg("buffer"),
            py::arg("copy") = false)
        .def_buffer(
            [](const memory_block &self)
            {
                using T = memory_block::value_type;

                return py::buffer_info{
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                    const_cast<T *>(self.data()), sizeof(T), "B", ssize(self), /*readonly=*/true
                };
            });
}

void
def_string(py::module_ &base)
{
    py::module_ m = base.def_submodule("string");

    py::class_<immutable_string>(m, "CString")
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
                return fmt::format("CString('{}')", self);
            })

        .def("bytes",
            [](const immutable_string &self)
            {
                return py::bytes(static_cast<std::string_view>(self));
            })

        .def("lstrip",
            [](const immutable_string &self)
            {
                return ltrim(self);
            })
        .def("rstrip",
            [](const immutable_string &self)
            {
                return rtrim(self);
            })
        .def("split",
            [](const immutable_string &self, std::optional<std::string_view> sep)
            {
                if (sep == std::nullopt)
                    throw std::invalid_argument("Default separator not implemented yet. You must provide a value for `sep`");
                auto sep_value = sep.value();
                if (sep_value.size() != 1) // TODO handle cases where separator is a string
                    throw std::invalid_argument("`sep` argument must be of size 1.");

                return self.split(sep_value[0]);
            },
            py::arg("sep") = std::nullopt
        )
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

}  // namespace

void
def_data(py::module_ &base)
{
    py::module_ m = base.def_submodule("data");

    def_data_pipeline(m);

    def_memory(m);

    def_string(m);

    def_text(m);
}

}  // namespace fairseq2
