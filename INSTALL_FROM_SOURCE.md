# Install from Source
The instructions in this document are for users who want to use fairseq2 on a
system for which no pre-built fairseq2 package is available, or for users who
want to work on the C++/CUDA code of fairseq2.

> [!NOTE]
> If you plan to edit and only modify Python portions of fairseq2, and if
> fairseq2 provides a pre-built nightly package for your system, we recommend
> using an editable pip installation as described in
> [Contribution Guidelines](CONTRIBUTING.md#setting-up-development-environment).


## 1. Clone the Repository
As first step, clone the fairseq2 Git repository to your machine:

```sh
git clone --recurse-submodules https://github.com/facebookresearch/fairseq2.git
```

Note the `--recurse-submodules` option that asks Git to clone the third-party
dependencies along with fairseq2. If you have already cloned fairseq2 without
`--recurse-submodules` before reading these instructions, you can run the
following command in your cloned repository to achieve the same effect:

```sh
git submodule update --init --recursive
```


## 2. Set up a Python Virtual Environment
In simplest case, you can run the following command to create an empty Python
virtual environment (shown for Python 3.8):

```sh
python3.8 -m venv ~/myvenv
```

And, activate it:

```sh
source ~/myvenv/bin/activate
```

You can check out the
[Python documentation](https://docs.python.org/3/library/venv.html#creating-virtual-environments)
to learn more about other environment options.

> [!IMPORTANT]
> We strongly recommend creating a new environment from scratch instead of
> reusing an existing one to avoid dependency conflicts.


> [!IMPORTANT]
> Manually building fairseq2 or any other C++ project in a Conda environment can
> become tricky and fail due to environment-specific conflicts with the host
> system libraries. Unless necessary, we recommend using a Python virtual
> environment to build fairseq2.


## 3. Install Dependencies

### 3.1 System Dependencies
fairseq2 depends on [libsndfile](https://github.com/libsndfile/libsndfile),
which can be installed via the system package manager on most Linux
distributions, or via Homebrew on macOS.

For Ubuntu-based systems, run:

```sh
sudo apt install libsndfile-dev
```

Similarly, on Fedora, run:

```sh
sudo dnf install libsndfile-devel
```

For other Linux distributions, please consult its documentation on how to
install packages.

For macOS, you can use Homebrew:

```sh
brew install libsndfile
```

### 3.2 PyTorch
Follow the instructions on [pytorch.org](https://pytorch.org/get-started/locally/)
to install the desired PyTorch version. Make sure that the version you install
is [supported](.#variants) by fairseq2.

### 3.3 CUDA
If you plan to build fairseq2 in a CUDA environment, you first have to install
a version of the CUDA Toolkit that matches the CUDA version of PyTorch. The
instructions for different toolkit versions can be found on NVIDIA’s website.

> [!NOTE]
> If you are on a compute cluster with `module` support (e.g. FAIR Cluster), you
> can typically activate a specific CUDA Toolkit version by
> `module load cuda/<VERSION>`.

### 3.4 pip
Finally, to install fairseq2’s C++ build dependencies (e.g. cmake, ninja), use:

```sh
pip install -r native/python/requirements-build.txt
```


## 4. Build fairseq2n

### CPU-Only Builds
The final step before installing fairseq2 is to build fairseq2n, fairseq2’s C++
library. Run the following command at the root directory of your repository to
configure the build:

```sh
cd native

cmake -GNinja -B build
```

Once the configuration step is complete, build fairseq2n using:

```sh
cmake --build build
```

fairseq2 uses reasonable defaults, so the command above is sufficient for a
standard installation; however, if you are familiar with CMake, you can check
out the advanced build options in
[`native/CMakeLists.txt`](native/CMakeLists.txt).

### CUDA Builds

> [!NOTE]
> If you are on a compute cluster with `module` support (e.g. FAIR Cluster), you
> can typically activate a specific CUDA Toolkit version by
> `module load cuda/<VERSION>`.

If you would like to build fairseq2’s CUDA kernels, set the `FAIRSEQ2N_USE_CUDA`
option `ON`. When turned on, the version of the CUDA Toolkit installed on your
machine and the version of CUDA that was used to build PyTorch must match:

```sh
cmake -GNinja -DFAIRSEQ2N_USE_CUDA=ON -B build
```

Similar to CPU-only build, follow this command with:

```sh
cmake --build build
```

### CUDA Architectures
By default, fairseq2 builds its CUDA kernels only for the Volta architecture.
You can override this setting using the `CMAKE_CUDA_ARCHITECTURES` option. For
instance, the following configuration generates binary and PTX codes for the
Ampere architecture (e.g. for A100):

```sh
cmake -GNinja -DCMAKE_CUDA_ARCHITECTURES="80-real;80-virtual" -DFAIRSEQ2N_USE_CUDA=ON -B build
```


## 5. Install fairseq2
Once you have built fairseq2n, the actual Python package installation is
straightforward. First install fairseq2n:

```sh
cd native/python

pip install .

cd -
```

Then, fairseq2:

```sh
pip install .
```

### Editable Install
In case you want to modify and test fairseq2, installing it in editable mode
will be more convenient:

```sh
cd native/python

pip install -e .

cd -

pip install -e .
```

Optionally, you can also install the development tools (e.g. linters,
formatters) if you plan to contribute to fairseq2. See
[Contribution Guidelines](CONTRIBUTING.md) for more information:

```sh
pip install -r requirements-devel.txt
```


## 6. Optional Sanity Check
To make sure that your installation has no issues, you can run the test suite:

```sh
pip install -r requirements-devel.txt

pytest
```

By default, the tests will be run on CPU; pass the `--device` (short form `-d`)
option to run them on a specific device (e.g. GPU):

```sh
pytest --device cuda:0
```
