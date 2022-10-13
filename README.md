<p align="center">
  <img src="doc/static/img/fairseq2_logo.png" width="150">
</p>

--------------------------------------------------------------------------------

[**Installation**](#installation) | [**Getting Started**](#getting-started) | [**Documentation**](#documentation)

fairseq2 is a sequence modeling toolkit that allows researchers and developers
to train custom models for translation, summarization, language modeling, and
other content generation tasks.

## Dependencies
fairseq2 versions corresponding to each PyTorch release:

| `fairseq2`   | `torch`     | `python`          |
| ------------ | ----------- | ----------------- |
| `main`       | `>=1.13.0`  | `>=3.8`, `<=3.10` |

## Installation
fairseq2 supports Linux and macOS operating systems. Please note though that
pre-built Conda and PyPI packages are *only* available for Linux. For
installation on macOS you can follow the instructions in the
[From Source](#from-source) section. At this time there are no plans to
introduce Windows support.

### Conda
coming soon...

### PyPI
coming soon...

## Installing From Source

### 1. Prerequisite: Install CMake
In order to build its C++ extension modules, fairseq2 requires
[CMake](https://cmake.org) 3.21 or newer. You can check whether your system has
a compatible version by running `cmake --version`. If the command is not found
or the output is less than 3.21, you should install a recent version.

Fortunately installing CMake is straightforward. If you are running Ubuntu >=
22.04, Fedora >= 35, or macOS with Homebrew, `[apt|dnf|brew] install cmake` is
all you have to do. Most other Linux distributions also have a fairly up-to-date
CMake version in their package repositories, so installing it boils down to
`<package_manager> install cmake`.

If your system does not provide a recent version, you can refer to the official
[installation instructions](https://cmake.org/download/). CMake already offers
installers for various operating systems. Although it requires a bit more work
than using a system package manager, it is still pretty straightforward to
install using an installer.

Lastly, if you don't have sudo access to your machine, or you don't want to
pollute your `/usr/bin`, you can install CMake locally. The officially endorsed
way for that is via [pipx](https://pypa.github.io/pipx/) (not to be confused
with pip), but Conda works equally well. Simply running
`[pipx|conda] install cmake` will install it to a local directory under your
home.

If you still have trouble, you can check out the
[Installing CMake](https://cliutils.gitlab.io/modern-cmake/chapters/intro/installing.html)
section of Modern CMake which has further instructions.

### 2. Optional Prerequisite: Install Ninja
[Ninja](https://ninja-build.org/) is a lightweight build system that focuses on
speed. It can significantly reduce build times for native (e.g. C++) code bases
compared to make and other traditional build systems. Although not required, we
recommend it for faster builds. Practically all system package managers offer
Ninja (e.g. `apt install ninja-build`), so it should be pretty straightforward
to install.

If you do not have sudo access to your machine, or you don’t want to pollute
your `/usr/bin`, similar to CMake, you can install it locally via
[pipx](https://pypa.github.io/pipx/) or Conda using
`[pipx|conda] install ninja`.

### 3. Clone the Repository
Once you have installed the prerequisite(s), clone the fairseq2 Git repository
to your machine:

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

### 4. Optional (Strongly Recommended): Set Up a Virtual Environment
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

### 5. Install PyTorch
Follow the instructions at [pytorch.org](https://pytorch.org/get-started) to
install PyTorch (in your preferred virtual environment). Note that fairseq2
supports only PyTorch 1.11 and later.

### 6. Build the C++ Extension Modules
The final step before installing fairseq2 is to build its C++ extension modules.
If you have Ninja installed (see step 2), run the following command at the root
directory of your repository to configure the build:

```
cmake -GNinja -B build
```

Otherwise, if you don’t have Ninja, run:

```
cmake -B build
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

### 7. Install the Package
Once you have built the extension modules, the actual Python package
installation is pretty straightforward:

```
pip install .
```

If you plan to play with the fairseq2 code, you can also install it in
[develop](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-e) (a.k.a.
editable) mode:

```
pip install -e .
```

Also remember that your contributions to fairseq2 are always welcome! Make sure
to check out our [guidelines](./CONTRIBUTING.md) to learn how you can help.

### 8. Optional Test
coming soon...

## Documentation
You can find our official documentation including tutorials and API reference
[here](https://fairinternal.github.io/fairseq2/nightly).

## Contributing
We always welcome your contributions to fairseq2! Please refer to our
[contribution guidelines](./CONTRIBUTING.md) to learn more about how to format,
test, and submit your work.

## License
This project is MIT licensed, as found in the [LICENSE](LICENSE) file. The
license applies to the pre-trained models as well.
