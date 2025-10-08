.. _tutorial-pudb:

==================================
:octicon:`bug` Debugging with PuDB
==================================


.. dropdown:: What you will learn
    :icon: multi-select
    :animate: fade-in

    * How to debug multi-node training using PuDB


.. dropdown:: Prerequisites
    :icon: multi-select
    :animate: fade-in

    * Get familiar with fairseq2 basics (:ref:`basics-design-philosophy`)

    * Ensure you have fairseq2 installed

    * Install PuDB: ``pip install pudb``

This tutorial explains how to debug your training sessions, including multi-node runs, using the `PuDB debugger <https://github.com/inducer/pudb>`.
PuDB is one of several remote debuggers you can use with fairseq2.

Placing the debugger breakpoint in the code
-------------------------------------------

Before setting a breakpoint, decide where in your code you want to start debugging.
Since fairseq2 supports multi-process training, ensure that the debugger is only invoked on the main process (rank 0) to prevent deadlocks.

Insert the following code where you want to set the breakpoint:

.. code-block:: python

    import os

    from fairseq2.utils.env import get_rank

    if get_rank(os.environ) == 0:
        from pudb.remote import set_trace

    set_trace(host="meta-fairseq2", port=6899, term_size=(80*3, 24*3), reverse=True)

**Explanation:**

- ``host="meta-fairseq2"``: Replace with the hostname accessible to both the machine running fairseq2 and your local machine.
- ``port=6899``: Choose an appropriate port that is open and not in use.
- ``term_size=(80*3, 24*3)``: Sets the terminal size for the debugger interface.
- ``reverse=True``: Instructs the debugger to initiate the connection from the host.


Initializing the socket for remote debugger
-------------------------------------------

On the host machine specified in the ``host`` parameter (`e.g.`, in our case it's ``meta-fairseq2``), run the following command to start listening on the specified port:

.. code-block:: bash

    stty -echo -icanon && nc -l -p 6899

.. note::

    - The command will appear to hang, which is expected as it's waiting for the debugger to connect.
    - Ensure that the chosen port (``6899`` in this case) is open and accessible.


Running fairseq2 with debugger
------------------------------

In the other terminal / pane you need to start the fairseq2 training as usual. Here we show an example using slurm cluster.

1. **Allocate Resources:**

    Obtain a compute allocation based on your cluster's configuration. Here's an example command using SLURM:

    .. code-block:: bash

        # Adjust the arguments (`--nodes`, `--ntasks-per-node`, etc.) as needed for your environment
        salloc --nodes=1 --ntasks-per-node=8 --cpus-per-task=10 -t 1:00:00 --gpus-per-node=8


2. **Start Training:**

    Launch your fairseq2 training job as you normally would. For example, for LLM training:

    .. code-block:: bash

        srun python -m recipes.lm.train $OUTPUT_DIR --config-file $CONFIG_YAML

3. **Connect to the Debugger:**

    Once the training reaches the breakpoint, the PuDB interface will appear in the terminal where you initialized the socket.

Example screenshot of the debugger:

.. image:: ../_static/img/tutorials/pudb.png
    :align: center
    :alt: PuDB example
    :width: 600

Please refer to the `PuDB docs and repo <https://github.com/inducer/pudb?tab=readme-ov-file#features>`_ to explore more features and familiarize yourself with the interface.
PuDB supports all standard ``pdb`` commands in the source view and offers additional functionality for an enhanced debugging experience.


Exiting the debugger
--------------------

Press ``q`` to quit the debugger.
This will terminate the socket session and stop the training job.
