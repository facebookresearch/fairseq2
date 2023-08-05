Install From Source (Conda)
===========================

1. Clone the Repository
-----------------------
As first step, clone the fairseq2 Git repository to your machine:

.. code-block::

    git clone --recurse-submodules https://github.com/fairinternal/fairseq2.git

Note the ``--recurse-submodules`` option that asks Git to clone the third-party
dependencies along with fairseq2. If you have already cloned fairseq2 without
``--recurse-submodules`` before reading these instructions, you can run the
following command in your cloned repository to achieve the same effect:

.. code-block::

    git submodule update --init --recursive


2. Set Up a Conda Environment
-----------------------------
In simplest case, you can run the following command to create an empty Conda
environment:

.. code-block::

    conda create --name myenv

and activate it:

.. code-block::

    conda activate myenv

Check out `the Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`_
to learn more about other environment options.

.. note::

    We strongly recommend creating a new environment from scratch instead of reusing
    an existing one to avoid package conflicts.


3. Install the Dependencies
---------------------------

.. note::

    We strongly recommend using the libmamba package solver in your Conda
    setup, which is significantly faster than Conda's legacy solver and can
    set up environments much quicker (i.e. the infamous "Solving dependencies"
    step). Follow the instructions `here <https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community>`_
    to set it up.

CPU-Only
^^^^^^^^
To install fairseq2's CPU-only runtime and build dependencies, use:

.. code-block::

    conda install -c pytorch -c conda-forge pytorch --file conda/requirements.txt

If you would like to tinker with fairseq2, use this alternative that also
installs the development tools:

.. code-block::

    conda install -c pytorch -c conda-forge pytorch\
        --file conda/requirements.txt\
        --file conda/requirements-devel.txt

Note that both commands will install the latest PyTorch version available in the
`pytorch` Conda channel. If you would like to install a specific version, append
a version specifier to ``pytorch`` in the commands above (e.g. ``pytorch==2.0.1``).

CUDA
^^^^
If you plan to build fairseq2 in a CUDA environment, you first have to install a
version of the CUDA Toolkit that matches the CUDA version of PyTorch. The
instructions for different toolkit versions can be found on NVIDIA's website.

.. note::

    If you are on a compute cluster with ``module`` support (e.g. FAIR Cluster),
    you can typically activate a specific CUDA Toolkit version by
    ``module load cuda/<VERSION>``.

Once you have the CUDA Toolkit installed, use the following command to install
fairseq2's runtime and build dependencies (shown for CUDA 11.7):

.. code-block::

    conda install -c pytorch -c nvidia -c conda-forge pytorch\
        --file conda/requirements.txt\
        --file conda/requirements-cu117.txt

Check out the `requirements-cuXX.txt` files in the `conda` directory to see the
supported CUDA versions.

Similar to CPU-only instructions, adding an extra ``--file conda/requirements-devel.txt``
to the command above will install the development tools as well.


4. Install the pip Dependencies
-------------------------------
Some of fairseq2's dependencies are only available in PyPI. Run the following
command to install them `after` you have installed the Conda packages as
described above:

.. code-block::

    pip install -r conda/requirements-pip.txt


5. Build the Extension Module
-----------------------------

CPU-Only Builds
^^^^^^^^^^^^^^^
The final step before installing fairseq2 is to build its C++ extension module.
Run the following command at the root directory of your repository to configure
the build:

.. code-block::

    cmake -GNinja -B build

Once the configuration step is complete, build the extension module using:

.. code-block::

    cmake --build build

fairseq2 uses reasonable defaults, so the command above is sufficient for a
standard installation; however, if you are familiar with CMake, you can check
out the advanced build options in CMakeLists.txt of the project.

CUDA Builds
^^^^^^^^^^^

.. note::

    If you are on a compute cluster with ``module`` support (e.g. FAIR Cluster),
    you can typically activate a specific CUDA Toolkit version by
    ``module load cuda/<VERSION>``.

If you would like to build fairseq2's CUDA kernels, set the ``FAIRSEQ2_USE_CUDA``
option ``ON``. When turned on, the version of the CUDA Toolkit installed on your
machine and the version of CUDA that was used to build PyTorch must match.

.. code-block::

    cmake -GNinja -DFAIRSEQ2_USE_CUDA=ON -B build

Similar to CPU-only build, follow this command with:

.. code-block::

    cmake --build build

CUDA Architectures
^^^^^^^^^^^^^^^^^^
By default, fairseq2 builds its CUDA kernels only for the Volta architecture.
You can override this setting using the ``CMAKE_CUDA_ARCHITECTURES``
`option <https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html>`_.
For instance, the following configuration generates binary and PTX codes
for the Ampere architecture (e.g. for A100).

.. code-block::

    cmake -GNinja -DCMAKE_CUDA_ARCHITECTURES="80-real;80-virtual" -DFAIRSEQ2_USE_CUDA=ON -B build

.. warning::

    Whenever you use fairseq2 within a new or updated Conda environment, make
    sure to delete its previous build artifacts by calling ``rm -rf build``.
    After deletion, you need to re-build its extension module to ensure that
    right dependencies are resolved.


6. Install the Package
----------------------
Once you have built the extension module, the actual Python package installation
is straightforward:

.. code-block::

    pip install --no-deps .

If you plan to play with fairseq2, you can also install it in
`editable <https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-e>`_ (a.k.a.
develop) mode:

.. code-block::

    pip install --no-deps -e .

Note the ``--no-deps`` option that is required in Conda environments.


7. Optional: Sanity Check
-------------------------
To make sure that your installation has no issues, you can run the Python tests:

.. code-block::

    pytest

By default, the tests will be run on CPU; optionally pass the ``--device``
(short form ``-d``) argument to run them on a specific device (e.g. NVIDIA GPU).

.. code-block::

    pytest --device cuda:0
