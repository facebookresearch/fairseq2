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
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <fairseq2/native/data/data.h>
#include <fairseq2/native/data/data_pipeline.h>
#include <fairseq2/native/data/data_processor.h>
#include <fairseq2/native/data/record_reader.h>
#include <fairseq2/native/data/stream.h>
#include <fairseq2/native/data/tape.h>

#include "fairseq2/native/extension/data/utils.h"

namespace py = pybind11;

using namespace fairseq2::detail;

namespace fairseq2 {
namespace detail {
namespace {

class data_pipeline_iterator {
public:
    explicit
    data_pipeline_iterator(data_pipeline &p) noexcept
      : pipeline_{&p}
    {}

    data
    next()
    {
        std::optional<data> d = pipeline_->next();
        if (!d)
            throw py::stop_iteration();

        return *std::move(d);
    }

private:
    data_pipeline *pipeline_;
};

}  // namespace
}  // namespace detail

void
def_data_pipeline(py::module_ &data_module)
{
    py::module_ m = data_module.def_submodule("data_pipeline");

    // DataPipeline
    py::class_<data_pipeline>(m, "DataPipeline")
        .def(py::init<>())

        .def(
            "__iter__",
            [](data_pipeline &self)
            {
                return data_pipeline_iterator{self};
            },
            py::keep_alive<0, 1>{})

        .def("reset", &data_pipeline::reset)

        .def_property_readonly("is_broken", &data_pipeline::is_broken)

        // state_dict
        .def(
            "state_dict",
            [](const data_pipeline &self)
            {
                tape t{};

                self.record_position(t);

                return py::dict{py::arg("position") = py::cast(t.storage())};
            })
        .def(
            "load_state_dict",
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

                std::vector<data> storage{};
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

        // Factories
        .def_static(
            "zip",
            [](
                std::vector<std::reference_wrapper<data_pipeline>> &refs,
                std::optional<std::vector<std::string>> names,
                bool warn_only,
                bool disable_parallelism)
            {
                std::vector<data_pipeline> pipelines{};

                pipelines.reserve(refs.size());

                std::transform(
                    refs.begin(), refs.end(), std::back_inserter(pipelines), [](auto &r) {
                        return std::move(r.get());
                    });

                return data_pipeline::zip(
                    std::move(pipelines), std::move(names), warn_only, disable_parallelism);
            },
            py::arg("pipelines"),
            py::arg("names") = std::nullopt,
            py::arg("warn_only") = false,
            py::arg("disable_parallelism") = false)
        .def_static(
            "round_robin",
            [](std::vector<std::reference_wrapper<data_pipeline>> &refs)
            {
                std::vector<data_pipeline> pipelines{};

                pipelines.reserve(refs.size());

                std::transform(
                    refs.begin(), refs.end(), std::back_inserter(pipelines), [](auto &r) {
                        return std::move(r.get());
                    });

                return data_pipeline::round_robin(std::move(pipelines));
            },
            py::arg("pipelines"));

    // DataPipelineIterator
    py::class_<data_pipeline_iterator>(m, "_DataPipelineIterator")
        .def(
            "__iter__",
            [](data_pipeline_iterator &self) -> data_pipeline_iterator &
            {
                return self;
            })
        .def("__next__", &data_pipeline_iterator::next);

    // DataPipelineError
    static py::exception<data_pipeline_error> py_data_pipeline_error{
        m, "DataPipelineError", PyExc_RuntimeError};

    // DataPipelineBuilder
    py::class_<data_pipeline_builder>(m, "DataPipelineBuilder")
        .def(
            "batch",
            [](
                data_pipeline_builder &self,
                std::size_t batch_size,
                bool drop_remainder) -> data_pipeline_builder &
            {
                self = std::move(self).batch(batch_size, drop_remainder);

                return self;
            },
            py::arg("batch_size"),
            py::arg("drop_remainder") = false)
        .def(
            "batch_by_length",
            [](
                data_pipeline_builder &self,
                std::vector<std::pair<std::size_t, std::size_t>> bucket_sizes,
                std::size_t max_seq_len,
                std::optional<std::string_view> selector,
                bool drop_remainder,
                bool warn_only) -> data_pipeline_builder &
            {
                self = std::move(self).batch_by_length(
                    std::move(bucket_sizes), max_seq_len, selector, drop_remainder, warn_only);

                return self;
            },
            py::arg("bucket_sizes"),
            py::arg("max_seq_len"),
            py::arg("selector") = std::nullopt,
            py::arg("drop_remainder") = false,
            py::arg("warn_only") = false)
        .def(
            "collate",
            [](
                data_pipeline_builder &self,
                std::optional<std::int32_t> pad_idx) -> data_pipeline_builder &
            {
                self = std::move(self).collate(pad_idx);

                return self;
            },
            py::arg("pad_idx") = std::nullopt)
        .def(
            "filter",
            [](data_pipeline_builder &self, predicate_fn &&f) -> data_pipeline_builder &
            {
                self = std::move(self).filter(std::move(f));

                return self;
            },
            py::arg("f"))
        .def(
            "map",
            [](
                data_pipeline_builder &self,
                const py::object &fn,
                std::optional<std::string_view> selector,
                std::size_t num_parallel_calls,
                bool warn_only) -> data_pipeline_builder &
            {
                self = std::move(self).map(
                    as_data_processor(fn, selector), num_parallel_calls, warn_only);

                return self;
            },
            py::arg("fn"),
            py::arg("selector") = std::nullopt,
            py::arg("num_parallel_calls") = 1,
            py::arg("warn_only") = false)
        .def(
            "prefetch",
            [](data_pipeline_builder &self, std::size_t num_examples) -> data_pipeline_builder &
            {
                self = std::move(self).prefetch(num_examples);

                return self;
            },
            py::arg("num_examples"))
        .def(
            "shard",
            [](
                data_pipeline_builder &self,
                std::size_t shard_idx,
                std::size_t num_shards) -> data_pipeline_builder &
            {
                self = std::move(self).shard(shard_idx, num_shards);

                return self;
            },
            py::arg("shard_idx"),
            py::arg("num_shards"))
        .def(
            "shuffle",
            [](
                data_pipeline_builder &self,
                std::size_t shuffle_window,
                bool strict,
                bool enabled) -> data_pipeline_builder &
            {
                self = std::move(self).shuffle(shuffle_window, strict, enabled);

                return self;
            },
            py::arg("shuffle_window"),
            py::arg("strict") = true,
            py::arg("enabled") = true)
        .def(
            "skip",
            [](data_pipeline_builder &self, std::size_t num_examples) -> data_pipeline_builder &
            {
                self = std::move(self).skip(num_examples);

                return self;
            },
            py::arg("num_examples"))
        .def(
            "take",
            [](data_pipeline_builder &self, std::size_t num_examples) -> data_pipeline_builder &
            {
                self = std::move(self).take(num_examples);

                return self;
            },
            py::arg("num_examples"))
        .def(
            "yield_from",
            [](data_pipeline_builder &self, yield_fn f) -> data_pipeline_builder &
            {
                self = std::move(self).yield_from(std::move(f));

                return self;
            },
            py::arg("f"))
        .def("and_return",
            [](data_pipeline_builder &self) -> data_pipeline
            {
                return std::move(self).and_return();
            });

    // DataProcessor
    py::class_<data_processor, std::shared_ptr<data_processor>>(m, "_DataProcessor")
        .def("__call__", &data_processor::process, py::call_guard<py::gil_scoped_release>{});

    // Factories
    m.def("list_files", &list_files, py::arg("pathname"), py::arg("pattern") = std::nullopt);

    m.def("read_sequence", &read_list, py::arg("seq"));

    m.def("read_zipped_records", &read_zipped_records, py::arg("pathname"));

    // TODO: Fix!
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

}  // namespace fairseq2
