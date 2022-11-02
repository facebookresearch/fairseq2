<p align="center">
  <img src="doc/static/img/logo.png" width="150"><br />
</p>

[![Release](https://github.com/fairinternal/fairseq2/actions/workflows/release.yaml/badge.svg)](https://github.com/fairinternal/fairseq2/actions/workflows/release.yaml)

--------------------------------------------------------------------------------

[**Getting Started**](#getting-started) | **Installing From: [Conda](#installing-from-conda), [PyPI](#installing-from-pypi), [Source](#installing-from-source)** | [**Contributing**](#contributing) | [**License**](#license)

fairseq2 is a sequence modeling toolkit that allows researchers and developers
to train custom models for translation, summarization, language modeling, and
other content generation tasks.

## Getting Started
You can find our full documentation including tutorials and API reference
[here](https://fairinternal.github.io/fairseq2/nightly)
([nightly](https://fairinternal.github.io/fairseq2/nightly)).

For recent changes, you can check out our [changelog](CHANGELOG.md).

fairseq2 mainly supports Linux. There is partial support for macOS with limited
feature set. Windows is not supported.

## Installing From Conda
coming soon...

## Installing From PyPI
coming soon...

## Installing From Source

### 1. Clone the Repository
As first step, clone the fairseq2 Git repository to your machine:

```
git clone --recurse-submodules https://github.com/fairinternal/fairseq2.git
```

Note the `--recurse-submodules` option that asks Git to clone the
[third-party](third-party) dependencies along with fairseq2. If you have already
cloned fairseq2 without `--recurse-submodules` before reading these
instructions, you can run the following command in your cloned repository to get
the same effect:

```
git submodule update --init --recursive
```

### 2. Optional (Strongly Recommended): Set Up a Virtual Environment
If you are not already in a Python virtual environment (e.g. Python `venv` or
Conda), we strongly recommend setting up one; otherwise, fairseq2 will be
installed to your user-wide or, if you have admin privileges, to the system-wide
Python package directory. In both cases, fairseq2 and its dependencies can cause
unintended compatibility issues with the system-provided Python packages.

Check out
[the official Python documentation](https://docs.python.org/3/library/venv.html#creating-virtual-environments)
to learn how to set up a virtual environment using the standard tooling. If you
prefer Conda, similar instructions can be found
[here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

### 3. Install the Build Dependencies
fairseq2 has a small set of prerequisites. You can install them (in your virtual
environment) via pip:

```
pip install -r requirements-build.txt
```

If you plan to play with or contribute to fairseq2, you should instead use:

```
pip install -r requirements-devel.txt
```

This second command will install linters, code formatters, and testing tools in
addition to build dependencies. Check out our
[contribution guidelines](./CONTRIBUTING.md) to learn how to use them.

### 4. Install PyTorch
Follow the instructions at [pytorch.org](https://pytorch.org/get-started) to
install PyTorch (in your virtual environment). Note that fairseq2 supports only
PyTorch 1.11 and later.

### 5. Build the C++ Extension Modules
The final step before installing fairseq2 is to build its C++ extension modules.
Run the following command at the root directory of your repository to configure
the build:

```
cmake -GNinja -B build
```

Once the configuration step is complete, build the extension modules using:

```
cmake --build build
```

fairseq2 uses reasonable defaults, so the command above is sufficient for a
standard installation; however, if you are familiar with CMake, you can check
out the advanced build options in [CMakeLists.txt](CMakeLists.txt).

**CUDA Builds**

By default, if the installed PyTorch has CUDA support, this will be inferred
during the configuration step and fairseq2 will be built using the same version
of CUDA. If, for any reason, you do not want to build fairseq2 with CUDA, you
can set the `FAIRSEQ2_USE_CUDA` option to `OFF` and disable the default
behavior:

```
cmake -GNinja -DFAIRSEQ2_USE_CUDA=OFF -B build
```

**CUDA Architectures**

By default, fairseq2 builds its CUDA kernels only for the Volta architecture.
You can override this setting using the
[`CMAKE_CUDA_ARCHITECTURES`](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html)
option. For instance, the following configuration generates binary and PTX codes
for the Ampere architecture (e.g. for A100).

```
cmake -GNinja -DCMAKE_CUDA_ARCHITECTURES=”80-real;80-virtual” -B build
```

### 6. Install the Package
Once you have built the extension modules, the actual Python package
installation is pretty straightforward:

```
pip install .
```

If you plan to play with fairseq2, you can also install it in
[develop](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-e) (a.k.a.
editable) mode:

```
pip install -e .
```

### 7. Optional: Sanity Check
To make sure that your installation has no issues, you can run the Python tests:

```
python -m tests
```

By default, the tests will be run on CPU; optionally pass the `--device` (short
form `-d`) argument to run them on a specific device (e.g. NVIDIA GPU).

```
python -m tests --device cuda:1
```

## Contributing
We always welcome contributions to fairseq2! Please refer to our
[contribution guidelines](./CONTRIBUTING.md) to learn how to format, test, and
submit your work.

## License
This project is MIT licensed, as found in the [LICENSE](LICENSE) file. The
license applies to the pre-trained models as well.

