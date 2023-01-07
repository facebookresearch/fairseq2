DataPipeline
============

.. currentmodule: fairseq2

.. class:: data.data_pipeline._DataPipeline
    :final:

    Bases: :class:`object`

    Lorem ipsum

    .. method:: next()

        Reads the next example in the data pipeline, or raises
        :class:`StopIteration` if the end of the data pipeline is reached.

        :rtype:
            typing.Any

        :raises StopIteration:
            Raised if the end of the data pipeline is reached.

    .. method:: skip(num_examples)

        Skips reading a specified number of examples.

        :param num_examples:
            The number of examples to skip.

        :type num_examples:
            int

        :returns:
            The number of examples skipped. It can be less than ``num_examples``
            if the end of the data pipeline is reached.
        :rtype:
            int

    .. method:: reset()

        Moves back to the first example.

    .. method:: record_position(t)

        Records the current position of the data pipeline to ``t``.

        :param t:
            The tape

        :type t:
            Tape

    .. method:: replay_position(t)

        Reloads the current position of the data pipeline from ``t``.

        :param t:
            The tape

        :type t:
            Tape

    .. property:: is_broken
        :type: bool

        Indicates whether the data pipeline is broken by a previous operation.
        If ``True``, any operation on this instance will raise a
        :class:`DataPipelineError`.
