// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/bindings/module.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <fairseq2n/exception.h>
#include <fairseq2n/data/byte_stream.h>
#include <fairseq2n/data/collater.h>
#include <fairseq2n/data/element_mapper.h>
#include <fairseq2n/data/data.h>
#include <fairseq2n/data/data_length_extractor.h>
#include <fairseq2n/data/data_pipeline.h>
#include <fairseq2n/data/file_mapper.h>
#include <fairseq2n/data/record_reader.h>
#include <fairseq2n/data/tape.h>
#include <fairseq2n/detail/exception.h>

namespace py = pybind11;

using namespace fairseq2n::detail;

namespace fairseq2n {
namespace detail {
namespace {

struct data_pipeline_deleter {
    void
    operator()(data_pipeline *pipeline) const
    {
        py::gil_scoped_release no_gil{};

        // By calling `reset()` here, we indirectly stop all daemon threads used
        // by `pipeline` without holding GIL. This way, we prevent any deadlocks
        // that might happen due to Python callbacks.
        try {
            pipeline->reset();
        } catch (const data_pipeline_error &) {}

        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        delete pipeline;
    }
};

// This class help us to gracefully delete data pipelines with active daemon
// threads (e.g. a prefetch op) during Python interpreter shutdown.
class data_pipeline_tracker {
    struct py_handle_hash {
        std::size_t
        operator()(const py::handle &h) const noexcept
        {
            return std::hash<void *>{}(h.ptr());
        }
    };

public:
    // Registers a hook with the `atexit` module to delete data pipelines that
    // are still alive during interpreter shutdown.
    void
    register_atexit_hook();

    // Delete `pipeline` if it is alive.
    void
    track(py::object pipeline);

private:
    void
    delete_alive_pipelines();

private:
    std::unordered_set<py::weakref, py_handle_hash> alive_pipelines_{};
};

void
data_pipeline_tracker::register_atexit_hook()
{
    py::module_ atexit_module = py::module_::import("atexit");

    auto hook = [this]
    {
        delete_alive_pipelines();
    };

    atexit_module.attr("register")(py::cpp_function{hook});
}

void
data_pipeline_tracker::track(py::object pipeline)
{
    // This `weakref` callback will be called when `pipeline` gets deleted
    // before interpreter shutdown. In such case, we just stop tracking it.
    auto remove_weakref = [this](const py::weakref &weakref)
    {
        alive_pipelines_.erase(weakref);
    };

    // We internally store a weak reference to `pipeline`. If it is still alive
    // by the time the interpreter is shutdown, we will use this weak reference
    // to get a handle to it.
    alive_pipelines_.emplace(std::move(pipeline), py::cpp_function{remove_weakref});
}

void
data_pipeline_tracker::delete_alive_pipelines()
{
    for (auto &weakref : alive_pipelines_) {
        py::object pipeline_obj = weakref();

        if (pipeline_obj.is_none())
            throw_<internal_error>(
                "One of the tracked data pipelines has already been deleted. Please file a bug report.");

        auto &pipeline = pipeline_obj.cast<data_pipeline &>();

        // A broken data pipeline does not have any active daemon threads.
        if (pipeline.is_broken())
            continue;

        {
            py::gil_scoped_release no_gil{};

            // By replacing with an empty one, we effectively delete the data
            // pipeline.
            pipeline = {};
        }
    }

    alive_pipelines_.clear();
}

data_pipeline_tracker &
data_pipeline_tracker() noexcept
{
    static class data_pipeline_tracker tracker{};

    return tracker;
}

// In extension modules, defining Python exception types with custom attributes
// is rather complicated; therefore, we use a thread-local variable to hold the
// example of the last failed pipeline operation.
thread_local std::optional<data> last_failed_example_;

class data_pipeline_iterator {
public:
    explicit
    data_pipeline_iterator(data_pipeline &pipeline) noexcept
      : pipeline_{&pipeline}
    {}

    data
    next()
    {
        std::optional<data> maybe_example{};

        {
            py::gil_scoped_release no_gil{};

            try {
                maybe_example = pipeline_->next();
            } catch (data_pipeline_error &ex) {
                last_failed_example_ = std::move(ex.maybe_example());

                throw;
            }

            // The operation was successful, clear the error state.
            last_failed_example_ = std::nullopt;
        }

        if (!maybe_example)
            throw py::stop_iteration();

        return *std::move(maybe_example);
    }

private:
    data_pipeline *pipeline_;
};

}  // namespace
}  // namespace detail

void
def_data_pipeline(py::module_ &data_module)
{
    data_pipeline_tracker().register_atexit_hook();

    py::module_ m = data_module.def_submodule("data_pipeline");

    // DataPipeline
    py::class_<data_pipeline, std::unique_ptr<data_pipeline, data_pipeline_deleter>>(
        m, "DataPipeline")
        .def(py::init<>())

        .def(
            "__iter__",
            [](data_pipeline &self)
            {
                return data_pipeline_iterator{self};
            },
            py::keep_alive<0, 1>{})

        .def("reset", &data_pipeline::reset, py::call_guard<py::gil_scoped_release>{})

        .def("is_infinite", &data_pipeline::is_infinite)

        .def_property_readonly("is_broken", &data_pipeline::is_broken)

        // state_dict
        .def(
            "state_dict",
            [](const data_pipeline &self)
            {
                tape t{};

                {
                    py::gil_scoped_release no_gil{};

                    self.record_position(t);
                }

                return py::dict{py::arg("position") = py::cast(t.storage())};
            })
        .def(
            "load_state_dict",
            [](data_pipeline &self, const py::dict &state_dict)
            {
                auto throw_invalid_arg = []()
                {
                    throw_<std::invalid_argument>(
                        "`state_dict` must contain a valid data pipeline state, but cannot be parsed as such.");
                };

                py::object value;
                try {
                    value = state_dict["position"];
                } catch (const py::error_already_set &ex) {
                    if (ex.matches(PyExc_KeyError))
                        throw_invalid_arg();

                    throw;
                }

                data_list storage{};
                try {
                    storage = value.cast<data_list>();
                } catch (const py::cast_error &) {
                    throw_invalid_arg();
                }

                tape t{std::move(storage)};

                try {
                    py::gil_scoped_release no_gil{};

                    self.reload_position(t);
                } catch (const std::invalid_argument &) {
                    throw_invalid_arg();
                }

                if (!t.is_eod())
                    throw_invalid_arg();
            },
            py::arg("state_dict"))

        // Factories
        .def_static(
            "concat",
            [](std::vector<std::reference_wrapper<data_pipeline>> &refs)
            {
                std::vector<data_pipeline> pipelines{};

                pipelines.reserve(refs.size());

                std::transform(
                    refs.begin(), refs.end(), std::back_inserter(pipelines), [](auto &r) {
                        return std::move(r.get());
                    });

                return data_pipeline::concat(std::move(pipelines));
            },
            py::arg("pipelines"))
        .def_static(
            "constant",
            [](data example, std::optional<std::string> key)
            {
                return data_pipeline::constant(std::move(example), std::move(key));
            },
            py::arg("example"),
            py::arg("key") = std::nullopt)
        .def_static(
            "count",
            [](std::int64_t start, std::int64_t step, std::optional<std::string> key)
            {
                return data_pipeline::count(start, step, std::move(key));
            },
            py::arg("start") = 0,
            py::arg("step") = 1,
            py::arg("key") = std::nullopt)
        .def_static(
            "round_robin",
            [](
                std::vector<std::reference_wrapper<data_pipeline>> &refs, bool stop_at_shortest)
            {
                std::vector<data_pipeline> pipelines{};

                pipelines.reserve(refs.size());

                std::transform(
                    refs.begin(), refs.end(), std::back_inserter(pipelines), [](auto &r) {
                        return std::move(r.get());
                    });

                return data_pipeline::round_robin(std::move(pipelines), stop_at_shortest);
            },
            py::arg("pipelines"),
            py::arg("stop_at_shortest") = false)
        .def_static(
            "sample",
            [](
                std::vector<std::reference_wrapper<data_pipeline>> &refs,
                std::optional<std::vector<float>> weights)
            {
                std::vector<data_pipeline> pipelines{};

                pipelines.reserve(refs.size());

                std::transform(
                    refs.begin(), refs.end(), std::back_inserter(pipelines), [](auto &r) {
                        return std::move(r.get());
                    });

                return data_pipeline::sample(std::move(pipelines), std::move(weights));
            },
            py::arg("pipelines"),
            py::arg("weights") = std::nullopt)
        .def_static(
            "zip",
            [](
                std::vector<std::reference_wrapper<data_pipeline>> &refs,
                std::optional<std::vector<std::string>> maybe_names,
                bool zip_to_shortest,
                bool flatten,
                bool disable_parallelism)
            {
                std::vector<data_pipeline> pipelines{};

                pipelines.reserve(refs.size());

                std::transform(
                    refs.begin(), refs.end(), std::back_inserter(pipelines), [](auto &r) {
                        return std::move(r.get());
                    });

                std::vector<std::string> names{};
                if (maybe_names)
                    names = *std::move(maybe_names);

                return data_pipeline::zip(
                    std::move(pipelines),
                    std::move(names),
                    zip_to_shortest,
                    flatten,
                    disable_parallelism);
            },
            py::arg("pipelines"),
            py::arg("names") = std::nullopt,
            py::arg("zip_to_shortest") = false,
            py::arg("flatten") = false,
            py::arg("disable_parallelism") = false);

    // DataPipelineIterator
    py::class_<data_pipeline_iterator>(m, "_DataPipelineIterator")
        .def(
            "__iter__",
            [](data_pipeline_iterator &self) -> data_pipeline_iterator &
            {
                return self;
            })
        .def("__next__", &data_pipeline_iterator::next);

    // DataPipelineBuilder
    py::class_<data_pipeline_builder>(m, "DataPipelineBuilder")
        .def(
            "bucket",
            [](
                data_pipeline_builder &self,
                std::size_t bucket_size,
                bool drop_remainder) -> data_pipeline_builder &
            {
                self = std::move(self).bucket(bucket_size, drop_remainder);

                return self;
            },
            py::arg("bucket_size"),
            py::arg("drop_remainder") = false)
        .def(
            "bucket_by_length",
            [](
                data_pipeline_builder &self,
                std::vector<std::pair<std::size_t, std::size_t>> bucket_sizes,
                std::optional<std::string> maybe_selector,
                bool skip_long_examples,
                bool drop_remainder) -> data_pipeline_builder &
            {
                self = std::move(self).bucket_by_length(
                    std::move(bucket_sizes),
                    data_length_extractor{std::move(maybe_selector)},
                    skip_long_examples,
                    drop_remainder);

                return self;
            },
            py::arg("bucket_sizes"),
            py::arg("selector") = std::nullopt,
            py::arg("skip_long_examples") = false,
            py::arg("drop_remainder") = false)
        .def(
            "collate",
            [](
                data_pipeline_builder &self,
                std::optional<std::int64_t> maybe_pad_value,
                std::int64_t pad_to_multiple,
                std::optional<std::vector<collate_options_override>> maybe_opt_overrides,
                std::size_t num_parallel_calls) -> data_pipeline_builder &
            {
                auto opts = collate_options()
                    .maybe_pad_value(maybe_pad_value).pad_to_multiple(pad_to_multiple);

                std::vector<collate_options_override> opt_overrides{};
                if (maybe_opt_overrides)
                    opt_overrides = *std::move(maybe_opt_overrides);

                map_fn f = collater(opts, std::move(opt_overrides));

                self = std::move(self).map(std::move(f), num_parallel_calls);

                return self;
            },
            py::arg("pad_value") = std::nullopt,
            py::arg("pad_to_multiple") = 1,
            py::arg("overrides") = std::nullopt,
            py::arg("num_parallel_calls") = 1)
        .def(
            "filter",
            [](data_pipeline_builder &self, predicate_fn fn) -> data_pipeline_builder &
            {
                self = std::move(self).filter(std::move(fn));

                return self;
            },
            py::arg("fn"))
        .def(
            "map",
            [](
                data_pipeline_builder &self,
                std::variant<map_fn, std::vector<map_fn>> fn,
                std::optional<std::string> maybe_selector,
                std::size_t num_parallel_calls) -> data_pipeline_builder &
            {
                map_fn f{};

                if (auto *map_functions = std::get_if<std::vector<map_fn>>(&fn))
                    // Combine all map functions in a single lambda and pass it
                    // to the C++ API.
                    f = [map_functions = std::move(*map_functions)](data &&example)
                    {
                        for (const map_fn &mf : map_functions)
                            example = mf(std::move(example));

                        return std::move(example);
                    };
                else
                    f = std::get<map_fn>(std::move(fn));

                element_mapper mapper{std::move(f), std::move(maybe_selector)};

                self = std::move(self).map(std::move(mapper), num_parallel_calls);

                return self;
            },
            py::arg("fn"),
            py::arg("selector") = std::nullopt,
            py::arg("num_parallel_calls") = 1)
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
            [](data_pipeline_builder &self, yield_fn fn) -> data_pipeline_builder &
            {
                self = std::move(self).yield_from(std::move(fn));

                return self;
            },
            py::arg("fn"))
        .def(
            "and_return",
            [](data_pipeline_builder &self, std::size_t max_num_warnings)
            {
                data_pipeline pipeline = std::move(self).and_return(max_num_warnings);

                py::object obj = py::cast(std::move(pipeline));

                // Ensure that the pipeline gets deleted during interpreter
                // shutdown if it is still alive.
                data_pipeline_tracker().track(obj);

                return obj;
            },
            py::arg("max_num_warnings") = 0);

    // DataPipelineError
    static py::exception<data_pipeline_error> py_data_pipeline_error{
        m, "DataPipelineError", PyExc_RuntimeError};

    m.def(
        "get_last_failed_example",
        []
        {
            return last_failed_example_;
        });


    // DataPipeline Factories
    m.def("list_files", &list_files, py::arg("pathname"), py::arg("pattern") = std::nullopt);

    m.def("read_sequence", &read_list, py::arg("seq"));

    m.def("read_zipped_records", &read_zipped_records, py::arg("pathname"));

    // Collater
    py::class_<collate_options_override>(m, "CollateOptionsOverride")
        .def(
            py::init([](
                std::string selector,
                std::optional<std::int64_t> maybe_pad_value,
                std::int64_t pad_to_multiple)
            {
                return collate_options_override{std::move(selector),
                    collate_options()
                        .maybe_pad_value(maybe_pad_value)
                        .pad_to_multiple(pad_to_multiple)};
            }),
            py::arg("selector"),
            py::arg("pad_value") = std::nullopt,
            py::arg("pad_to_multiple") = 1)
        .def_property_readonly(
            "selector",
            [](const collate_options_override &self)
            {
                return self.selector().string_();
            })
        .def_property_readonly(
            "pad_value",
            [](const collate_options_override &self)
            {
                return self.options().maybe_pad_value();
            })
        .def_property_readonly(
            "pad_to_multiple",
            [](const collate_options_override &self)
            {
                return self.options().pad_to_multiple();
            });

    py::class_<collater, std::shared_ptr<collater>>(m, "Collater")
        .def(
            py::init([](
                std::optional<std::int64_t> maybe_pad_value,
                std::int64_t pad_to_multiple,
                std::optional<std::vector<collate_options_override>> maybe_opt_overrides)
            {
                auto opts = collate_options()
                    .maybe_pad_value(maybe_pad_value).pad_to_multiple(pad_to_multiple);

                std::vector<collate_options_override> opt_overrides{};
                if (maybe_opt_overrides)
                    opt_overrides = *std::move(maybe_opt_overrides);

                return std::make_shared<collater>(opts, std::move(opt_overrides));
            }),
            py::arg("pad_value") = std::nullopt,
            py::arg("pad_to_multiple") = 1,
            py::arg("overrides") = std::nullopt)
        .def("__call__", &collater::operator(), py::call_guard<py::gil_scoped_release>{});

    map_functors().register_<collater>();

    // FileMapper
    py::class_<file_mapper, std::shared_ptr<file_mapper>>(m, "FileMapper")
        .def(
            py::init<std::optional<std::string>, std::optional<std::size_t>>(),
            py::arg("root_dir") = std::nullopt,
            py::arg("cached_fd_count") = std::nullopt)
        .def("__call__", &file_mapper::operator(), py::call_guard<py::gil_scoped_release>{});

    map_functors().register_<file_mapper>();

    // RecordError
    static py::exception<record_error> py_record_error{m, "RecordError", PyExc_RuntimeError};

    // ByteStreamError
    static py::exception<byte_stream_error> py_byte_stream_error{
        m, "ByteStreamError", PyExc_RuntimeError};

    // TODO: Remove once https://github.com/pybind/pybind11/pull/4366 lands.
    py::register_exception_translator([](std::exception_ptr ptr)
    {
        if (!ptr)
            return;

        auto raise_error = [&ptr](const std::exception &e, const py::object &err) {
            py::detail::handle_nested_exception(e, ptr);

            py::detail::raise_err(err.ptr(), e.what());
        };

        try {
            std::rethrow_exception(ptr);
        } catch (const byte_stream_error &e) {
            raise_error(e, py_byte_stream_error);
        } catch (const record_error &e) {
            raise_error(e, py_record_error);
        } catch (const data_pipeline_error &e) {
            raise_error(e, py_data_pipeline_error);
        }
    });
}

}  // namespace fairseq2n
