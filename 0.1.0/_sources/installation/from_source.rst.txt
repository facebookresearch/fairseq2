Install From Source (C++/CUDA)
==============================


1. Clone the Repository
-----------------------
As first step, clone the fairseq2 Git repository to your machine:

.. code-block::

    git clone --recurse-submodules https://github.com/facebookresearch/fairseq2.git

Note the ``--recurse-submodules`` option that asks Git to clone the third-party
dependencies along with fairseq2. If you have already cloned fairseq2 without
``--recurse-submodules`` before reading these instructions, you can run the
following command in your cloned repository to achieve the same effect:

.. code-block::

    git submodule update --init --recursive


2. Set Up a Python Virtual Environment
--------------------------------------
In simplest case, you can run the following command to create an empty virtual
environment (shown for Python 3.8):

.. code-block::

    python3.8 -m venv ~/myvenv

and activate it:

.. code-block::

    source ~/myvenv/bin/activate

Check out `the Python documentation <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`_
to learn mode about other environment options.

.. note::

    We strongly recommend creating a new environment from scratch instead of reusing
    an existing one to avoid package conflicts.


3. Install the Dependencies
---------------------------

System
^^^^^^
fairseq2 has dependency on some system libraries that can be typically installed
via native package managers. For Ubuntu-based systems, run:

.. code-block::

    sudo apt install libsndfile-dev

Similarly, on Fedora run:

.. code-block::

    sudo dnf install libsndfile-devel

CUDA
^^^^
If you plan to build fairseq2 in a CUDA environment, you first have to install a
version of the CUDA Toolkit that matches the CUDA version of PyTorch. The
instructions for different toolkit versions can be found on NVIDIA's website.

.. note::

    If you are on a compute cluster with ``module`` support (e.g. FAIR Cluster),
    you can typically activate a specific CUDA Toolkit version by
    ``module load cuda/<VERSION>``.

pip
^^^
To install fairseq2's build dependencies, use:

.. code-block::

    pip install torch -r fairseq2n/python/requirements-build.txt


4. Build the Extension Module
-----------------------------

CPU-Only Builds
^^^^^^^^^^^^^^^
The final step before installing fairseq2 is to build its C++ extension module.
Run the following command at the root directory of your repository to configure
the build:

.. code-block::

    cd fairseq2n

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

If you would like to build fairseq2's CUDA kernels, set the ``FAIRSEQ2N_USE_CUDA``
option ``ON``. When turned on, the version of the CUDA Toolkit installed on your
machine and the version of CUDA that was used to build PyTorch must match.

.. code-block::

    cmake -GNinja -DFAIRSEQ2N_USE_CUDA=ON -B build

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

    cmake -GNinja -DCMAKE_CUDA_ARCHITECTURES="80-real;80-virtual" -DFAIRSEQ2N_USE_CUDA=ON -B build


5. Install the Package
----------------------
Once you have built the extension module, the actual Python package installation
is straightforward. First install fairseq2n:

.. code-block::

    cd fairseq2n/python

    pip install -e .

    cd ../..

And then, fairseq2:

.. code-block::

    FAIRSEQ2N_DEVEL=1 pip install -e .


6. Optional: Sanity Check
-------------------------
To make sure that your installation has no issues, you can run the Python tests:

.. code-block::

    pytest

By default, the tests will be run on CPU; optionally pass the ``--device``
(short form ``-d``) argument to run them on a specific device (e.g. NVIDIA GPU).

.. code-block::

    pytest --device cuda:0
