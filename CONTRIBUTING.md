# Contributing to fairseq2
We want to make contributing to fairseq2 as easy and transparent as possible.
Please make sure to read this guideline carefully.


## Setting up Development Environment
fairseq2 consists of two packages; the user-facing fairseq2 package implemented
in pure Python, and the fairseq2n package that contains the C++ and CUDA
portions of the library. If fairseq2n is available as a pre-built package for
your system (check installation instructions for your operating system in
[README](.)), and if you are interested in only modifying Python portions of
fairseq2, you can use an editable pip installation as described below.
Otherwise, if you are planning to work on C++/CUDA, or if fairseq2n is not
available as a pre-built package for your system, please follow the install
instructions [here](INSTALL_FROM_SOURCE.md).

For an editable installation, first, clone the fairseq2 repository to your
machine.

```sh
git clone https://github.com/facebookresearch/fairseq2.git

cd fairseq2
```

Then, install the fairseq2 package in editable mode.

```sh
pip install -e .
```

Finally, make sure to install the development tools (e.g. linters and
formatters).

```sh
pip install -r requirements-devel.txt
```


## Testing Your Work

### Python and C++/CUDA
Any work that you plan to contribute should ideally be covered by a unit or
integration test. Once you have all your tests in place, ensure the full test
suite passes.

```sh
pytest
```

By default, the tests will be run on CPU; pass the `--device` (short form `-d`)
option to run them on a specific device (e.g. GPU).

```sh
pytest --device cuda:0
```

### C++/CUDA
If you have changes in C++ or CUDA, in addition to `pytest`, also run the native
tests.

```sh
fairseq2n/build/tests/run-tests
```


## Documenting Your Work
Any new or revised user-facing feature included in your work should have an
accompanying documentation. Depending on the scope of the work, the
documentation can be just docstrings in Python code, or, for larger features,
one or more Sphinx RST files. For docstrings, make sure to follow our formatting
conventions. You can check out any Python file in our code base to study how we
format our docstrings.

To build and test out the library documentation, run the following commands.

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
If you have made changes to the Python code, run the following commands and
address any issues reported.

```sh
mypy . && flake8 .
```

### C++/CUDA
If you have touched C++ or CUDA files, lint your code with an up-to-date version
of the clang toolkit and address any issues reported.

```sh
cd fairseq2n

CC=clang CXX=clang++ cmake -GNinja -DFAIRSEQ2N_RUN_CLANG_TIDY=ON -B build

cmake --build build
```

Alternatively:

```sh
cd fairseq2n

CC=clang CXX=clang++ cmake -GNinja -B build

run-clang-tidy -p build
```


## Formatting Your Work

### Python
For Python code, run the following commands.

```sh
isort . && black .
```

### C++/CUDA
For C++, we do not enforce our coding conventions via a tool (e.g.
clang-format), but we expect you to follow them. You can check out any C++ file
in our code base to study our conventions. Since C++ syntax can become pretty
complex at times, refrain from being too pedantic and prioritize readability
over convention.


## Check List For Pull Requests
1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests, and ensure the entire
   test suite passes.
3. If you've added or revised a user-facing feature, update the documentation.
4. Lint and format your code.
5. If you haven't already, complete the Contributor License Agreement ("CLA").


## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>


## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.


## License
By contributing to fairseq2, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
