.. _installation_from_source:

:octicon:`file-binary` Installing from Source
=============================================

The instructions in this document are for users who want to use fairseq2 on a
system for which no pre-built fairseq2 package is available, or for users who
want to work on the C++/CUDA code of fairseq2.

.. note::
   If you plan to edit and only modify Python portions of fairseq2, and if
   fairseq2 provides a pre-built nightly package for your system, we recommend
   using an editable pip installation as described in
   :ref:`faq-contributing-setup`.

1. Clone the Repository
-----------------------

As first step, clone the fairseq2 Git repository to your machine:

.. code-block:: sh

    git clone --recurse-submodules https://github.com/facebookresearch/fairseq2.git

Note the ``--recurse-submodules`` option that asks Git to clone the third-party
dependencies along with fairseq2. If you have already cloned fairseq2 without
``--recurse-submodules`` before reading these instructions, you can run the
following command in your cloned repository to achieve the same effect:

.. code-block:: sh

    git submodule update --init --recursive

2. Set up a Python Virtual Environment
--------------------------------------

In simplest case, you can run the following command to create an empty Python
virtual environment (shown for Python 3.8):

.. code-block:: sh

    python3.8 -m venv ~/myvenv

And, activate it:

.. code-block:: sh

    source ~/myvenv/bin/activate

You can check out the
`Python documentation <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`_
to learn more about other environment options.

.. important::
   We strongly recommend creating a new environment from scratch instead of
   reusing an existing one to avoid dependency conflicts.

.. important::
   Manually building fairseq2 or any other C++ project in a Conda environment can
   become tricky and fail due to environment-specific conflicts with the host
   system libraries. Unless necessary, we recommend using a Python virtual
   environment to build fairseq2.

3. Install Dependencies
-----------------------

3.1 System Dependencies
^^^^^^^^^^^^^^^^^^^^^^^

fairseq2 depends on `libsndfile <https://github.com/libsndfile/libsndfile>`__,
which can be installed via the system package manager on most Linux
distributions, or via Homebrew on macOS.

For Ubuntu-based systems, run:

.. code-block:: sh

    sudo apt install libsndfile-dev

Similarly, on Fedora, run:

.. code-block:: sh

    sudo dnf install libsndfile-devel

For other Linux distributions, please consult its documentation on how to
install packages.

For macOS, you can use Homebrew:

.. code-block:: sh

    brew install libsndfile

3.2 PyTorch
^^^^^^^^^^^

Follow the instructions on `pytorch.org <https://pytorch.org/get-started/locally/>`_
to install the desired PyTorch version. Make sure that the version you install
is supported by fairseq2.

3.3 CUDA
^^^^^^^^

If you plan to build fairseq2 in a CUDA environment, you first have to install
a version of the CUDA Toolkit that matches the CUDA version of PyTorch. The
instructions for different toolkit versions can be found on NVIDIA’s website.

.. note::
   If you are on a compute cluster with ``module`` support (e.g. FAIR Cluster),
   you can typically activate a specific CUDA Toolkit version by
   ``module load cuda/<VERSION>``.

3.4 pip
^^^^^^^

Finally, to install fairseq2’s C++ build dependencies (e.g. cmake, ninja), use:

.. code-block:: sh

    pip install -r native/python/requirements-build.txt

4. Build fairseq2n
------------------

4.1 CPU-Only Builds
^^^^^^^^^^^^^^^^^^^

The final step before installing fairseq2 is to build fairseq2n, fairseq2’s C++
library. Run the following command at the root directory of your repository to
configure the build:

.. code-block:: sh

    cd native

    cmake -GNinja -B build

Once the configuration step is complete, build fairseq2n using:

.. code-block:: sh

    cmake --build build

fairseq2 uses reasonable defaults, so the command above is sufficient for a
standard installation; however, if you are familiar with CMake, you can check
out the advanced build options in
`native/CMakeLists.txt <https://github.com/facebookresearch/fairseq2/blob/main/native/CMakeLists.txt>`__.

4.2 CUDA Builds
^^^^^^^^^^^^^^^
.. note::
   If you are on a compute cluster with ``module`` support (e.g. FAIR Cluster),
   you can typically activate a specific CUDA Toolkit version by
   ``module load cuda/<VERSION>``.

If you would like to build fairseq2’s CUDA kernels, set the ``FAIRSEQ2N_USE_CUDA``
option to ``ON``. When turned on, the version of the CUDA Toolkit installed on
your machine and the version of CUDA that was used to build PyTorch must match:

.. code-block:: sh

    cmake -GNinja -DFAIRSEQ2N_USE_CUDA=ON -B build

Similar to CPU-only build, follow this command with:

.. code-block:: sh

    cmake --build build

4.3 CUDA Architectures
^^^^^^^^^^^^^^^^^^^^^^

By default, fairseq2 builds its CUDA kernels only for the Volta architecture.
You can override this setting using the ``CMAKE_CUDA_ARCHITECTURES`` option.
For
instance, the following configuration generates binary and PTX codes for the
Ampere architecture (e.g. for A100):

.. code-block:: sh

    cmake -GNinja -DCMAKE_CUDA_ARCHITECTURES="80-real;80-virtual" -DFAIRSEQ2N_USE_CUDA=ON -B build

5. Install fairseq2
-------------------

Once you have built fairseq2n, the actual Python package installation is
straightforward. First install fairseq2n:

.. code-block:: sh

    cd native/python

    pip install .

    cd -

Then, fairseq2:

.. code-block:: sh

    pip install .

5.1 Editable Install
^^^^^^^^^^^^^^^^^^^^

In case you want to modify and test fairseq2, installing it in editable mode
will be more convenient:

.. code-block:: sh

    cd native/python

    pip install -e .

    cd -

    pip install -e .

Optionally, you can also install the development tools (e.g. linters,
formatters) if you plan to contribute to fairseq2. See
:ref:`faq-contributing` for more information:

.. code-block:: sh

    pip install -r requirements-devel.txt

6. Optional Sanity Check
^^^^^^^^^^^^^^^^^^^^^^^^

To make sure that your installation has no issues, you can run the test suite:

.. code-block:: sh

    pip install -r requirements-devel.txt

    pytest

By default, the tests will be run on CPU; pass the ``--device`` (short form
``-d``) option to run them on a specific device (e.g. GPU):

.. code-block:: sh

    pytest --device cuda:0

7. Example End-to-End Build Script

.. code-block:: sh

    python_version=3.10.14
    torch_version=2.7.1
    cuda_version=12.8
    variant=cu128
    arch=linux-x86_64

    conda create --name fs2_build_wheel python=$python_version
    conda activate fs2_build_wheel
    conda install -c conda-forge libsndfile compilers=1.2.0

    pip install --extra-index-url https://download.pytorch.org/whl/$variant\
                torch==$torch_version

    git clone --recurse-submodules git@github.com:facebookresearch/fairseq2.git
    cd fairseq2
    pip install --requirement native/python/requirements-build.txt

    version=0.5.1
    tools/set-project-version.sh --native-only $version+$variant

    cuda_archs="70-real;80-real;80-virtual"
    cuda=ON
    build_type=Release
    lto=ON
    SANITIZERS=nosan

    # fairseq2/native
    cd native
    cmake\
    -GNinja\
    -DCMAKE_BUILD_TYPE=$build_type\
    -DCMAKE_CUDA_ARCHITECTURES="$cuda_archs"\
    -DFAIRSEQ2N_PERFORM_LTO=$lto\
    -DFAIRSEQ2N_SANITIZERS="${SANITIZERS/_/;}"\
    -DFAIRSEQ2N_TREAT_WARNINGS_AS_ERRORS=ON\
    -DFAIRSEQ2N_USE_CUDA=$cuda\
    -DFAIRSEQ2N_PYTHON_DEVEL=OFF\
    -B build

    cmake --build build

    # fairseq2/native/python
    cd python
    pip wheel\
    --use-pep517\
    --no-build-isolation\
    --no-deps\
    --config-settings "--build-option=--plat-name"\
    --config-settings "--build-option=manylinux_2_28_$arch"\
    --wheel-dir build/wheelhouse\
    .

    ls build/wheelhouse/

    # fairseq2/
    cd ../../
    pip wheel --no-deps --wheel-dir build/wheelhouse .
    ls build/wheelhouse/
