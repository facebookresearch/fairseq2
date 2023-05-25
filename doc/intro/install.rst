.. _installation:

Installation
============

.. highlight:: bash

Installation from pip
---------------------

.. TODO:: publish wheels to pip

Default installation from source
--------------------------------

This is the preferred installation to contribute to fairseq2
because it installs development dependencies.

By default ``make install`` installs PyTorch and fairseq2 in a new venv_.

.. note::
  Even though we tried to make this installation script robust to Ubuntu/Fedora/MacOs differences
  there might be specific issues depending on your environment.
  If that's the case we advice you to follow up the step by step instructions,
  where it would be easier to pin point the issue.

If you want to reuse an existing venv, you can use ``make install VENV=/path/to/venv/bin/``.
If you have a Conda_ environment activated, ``make install`` installs fairseq2 inside it.
If you provide a venv or a Conda environment,
``make install`` assumes you've already installed PyTorch inside, and won't try to upgrade it.
Run ``make tests`` to make sure your installation is correct.

.. note:: If you modify the C++/Python API, you'll need to rerun ``make install``.
  If you simply modify the C++ code but not the Python API, ``make build`` is enough.
  For C++ development it can also be useful to replace the release mode of fairseq2 to Debug,
  by editing the ``Makefile``.

.. _venv: https://docs.python.org/3/library/venv.html
.. _Conda: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands


Step by Step instructions and special configurations
----------------------------------------------------


1. Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~

As first step, clone the fairseq2 Git repository to your machine::

  git clone --recurse-submodules https://github.com/fairinternal/fairseq2.git

Note the ``--recurse-submodules`` option that asks Git to clone the
third-party dependencies along with fairseq2. If you have already
cloned fairseq2 without ``--recurse-submodules`` before reading these
instructions, you can run the following command in your cloned repository to
achieve the same effect::

  git submodule update --init --recursive

2. Set up a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you aren't already in a Python virtual environment
(like venv_ or Conda_),
we strongly recommend setting up one.
Otherwise fairseq2 will be installed to your user-wide
or, if you have administrator privileges,
to the system-wide Python package directory.
In both cases fairseq2 and its dependencies can cause
unintended compatibility issues with the system-provided Python packages.

Check out
`the official Python documentation <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`_
to learn how to set up a virtual environment using the standard tooling.
If you prefer `Conda`_ follow their instructions.

3. Install build dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fairseq2 has a small set of prerequisites. You can install them (in your virtual
environment) via pip::

  pip install -r requirements-build.txt

If you plan to play with or contribute to fairseq2, you should instead use::

  pip install -r requirements-devel.txt

This second command installs linters,
code formatters, and testing tools in addition to build dependencies.
`Check out the contribution guidelines. <https://github.com/fairinternal/fairseq2/blob/main/CONTRIBUTING.md>`_
to learn how to use them.

.. TODO:: move contributing.md to contributing.rst

4. Install PyTorch
~~~~~~~~~~~~~~~~~~

Follow the instructions at `pytorch.org <https://pytorch.org/get-started>`_
to install PyTorch (in your virtual environment).
Note that fairseq2 supports only PyTorch 1.12.1 and later.

5. Build the C++ extension module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The final step before installing fairseq2 is to build its C++ extension module.
Run the following command at the root directory of your repository to configure
the build::

  cmake -GNinja -B build

Once the configuration step is complete, build the extension module using::

  cmake --build build

fairseq2 uses reasonable defaults,
so the preceding command is sufficient for a standard installation;
however, if you are familiar with CMake,
you can look up the advanced build options in ``CMakeLists.txt``.

**CUDA builds**

If you would like to build fairseq2's CUDA kernels, set the ``FAIRSEQ2_USE_CUDA``
option ``ON``. When turned on, the version of the CUDA Toolkit and the version of
CUDA that was used to build PyTorch must match.::

  cmake -GNinja -DFAIRSEQ2_USE_CUDA=ON -B build

If you are on a compute cluster with ``module`` support (e.g. FAIR Cluster), you
can typically activate a specific CUDA Toolkit version by
``module load cuda/<VERSION>``.

**CUDA architectures**

By default, fairseq2 builds its CUDA kernels only for the Volta architecture.
You can override this setting using the
`CMAKE_CUDA_ARCHITECTURES <https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html>`_
option. For instance, the following configuration generates binary and PTX codes
for the Ampere architecture (e.g. for A100).::

  cmake -GNinja -DCMAKE_CUDA_ARCHITECTURES="80-real;80-virtual" -DFAIRSEQ2_USE_CUDA=ON -B build

6. Install the package
~~~~~~~~~~~~~~~~~~~~~~

Once you have built the extension module, the actual Python package installation
is pretty straightforward::

  pip install .

If you plan to play with fairseq2,
you can also install it in
`editable mode <https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-e>`
(also know as develop mode)::

  pip install -e .

7. Sanity check
~~~~~~~~~~~~~~~

To make sure that your installation has no issues,
run the Python tests::

  pytest
