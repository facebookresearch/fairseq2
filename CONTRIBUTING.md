# Contributing to fairseq2
We want to make contributing to fairseq2 as easy as possible. Please make sure
to read this guideline carefully.


## Setting up Development Environment
fairseq2 consists of two packages; the user-facing fairseq2 package implemented
in pure Python, and the fairseq2n package that contains the C++ and CUDA
portions of the library. If pre-built fairseq2n nightly packages are available
for your system (check [README](.#nightlies)), and if you are interested in only
modifying Python portions of fairseq2, you can use an editable pip installation
as described below. Otherwise, if you are planning to work on C++ or CUDA, or if
fairseq2n is not available as a pre-built package for your system, please follow
the installation instructions [here](INSTALL_FROM_SOURCE.md).

For an editable installation, first, install a nightly build of fairseq2n (shown
for PyTorch `2.2.0` and variant `cu118`):

```sh
pip install fairseq2n\
  --pre --upgrade --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.2.0/cu118
```

> [!WARNING]
> fairseq2n relies on the C++ API of PyTorch which has no API/ABI compatibility
> between releases. This means **you have to install the fairseq2n variant that
> exactly matches your PyTorch version**. Otherwise, you might experience issues
> like immediate process crashes or spurious segfaults. For the same reason, if
> you upgrade your PyTorch version, you must also upgrade your fairseq2n
> installation.

Then, clone the fairseq2 repository to your machine:

```sh
git clone https://github.com/facebookresearch/fairseq2.git

cd fairseq2
```

And, install the fairseq2 package in editable mode:

```sh
pip install -e .
```

Finally, make sure to install the development tools (e.g. linters and
formatters):

```sh
pip install -r requirements-devel.txt
```

> [!NOTE]
> Any time you pull the latest fairseq2 commits from GitHub, make sure to re-run
> the fairseq2n installation command above to get the most up-to-date binary. If
> you observe runtime or test failures after the installation, it might be
> because the latest nightlies are not published yet. If the problem persists
> for more than 12 hours, please create a
> [GitHub issue](https://github.com/facebookresearch/fairseq2/issues/new/choose).

## Testing Your Work

### Python and C++
Any work that you plan to contribute should ideally be covered by a unit or
integration test. Once you have all your tests in place, ensure the full test
suite passes:

```sh
pytest
```

By default, the tests will be run on CPU; pass the `--device` (short form `-d`)
option to run them on a specific device (e.g. GPU):

```sh
pytest --device cuda:0
```

### C++
If you have changes in C++ or CUDA, in addition to `pytest`, also run the native
tests:

```sh
native/build/tests/run-tests
```


## Documenting Your Work
Any new or revised user-facing feature included in your work should have an
accompanying documentation. Depending on the scope of the work, the
documentation can be just docstrings in Python code, or, for larger features,
one or more Sphinx RST files. For docstrings, make sure to follow our formatting
conventions. You can check out any Python file in our code base to study how we
format our docstrings.

To build and test out the library documentation, run the following commands:

```sh
cd doc

pip install -r requirements.txt

make html

cd build/html

python -m http.server 8084
```

and, visit http://localhost:8084 in your browser.


## Linting Your Work

### Python
If you have made changes to the Python code, run the following command and
address any issues reported:

```sh
mypy && flake8 .
```

### C++
If you have touched C++ or CUDA files, lint your code with an up-to-date version
of the clang toolkit and address any issues reported:

```sh
cd native

CC=clang CXX=clang++ cmake -GNinja -DFAIRSEQ2N_RUN_CLANG_TIDY=ON -B build

cmake --build build
```

Alternatively:

```sh
cd native

CC=clang CXX=clang++ cmake -GNinja -B build

run-clang-tidy -p build
```


## Formatting Your Work

### Python
For Python code, run the following command:

```sh
isort . && black .
```

### C++
For C++ and CUDA, we do not enforce our coding conventions via a tool (e.g.
clang-format), but we expect you to follow them. You can check out any C++ file
in our code base to study our conventions. Since C++ syntax can become pretty
complex at times, refrain from being too pedantic and prioritize readability
over convention.


## Check List for Pull Requests
1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests, and ensure the entire
   test suite passes.
3. If you've added or revised a user-facing feature, update the documentation.
4. Lint and format your code.
5. If you haven't already, complete the Contributor License Agreement ("CLA").


## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open-source projects.

Complete your CLA here: <https://code.facebook.com/cla>


## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.


## License
By contributing to fairseq2, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
