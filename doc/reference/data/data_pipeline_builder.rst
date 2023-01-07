DataPipelineBuilder
===================

.. currentmodule: fairseq2

.. class:: data.data_pipeline._DataPipelineBuilder
    :final:

    Bases: :class:`object`

    Lorem ipsum

    .. method:: batch(batch_size, drop_remainder)

        Combines a number of consecutive examples into a single example.

        :param batch_size:
            The number of examples to combine.
        :param drop_remainder:
            Indicates whether to drop the last batch in the case it has fewer
            than ``batch_size`` examples.

        :type batch_size:
            int
        :type drop_remainder:
            bool

        :rtype:
            DataPipelineBuilder

    .. method:: map(fn)

        Applies ``fn`` to every example in the data pipeline.

        :param fn:
            The function to apply.

        :type fn:
            ~typing.Callable[[~typing.Any], ~typing.Any]

        :rtype:
            DataPipelineBuilder

    .. method:: shard(shard_idx, num_shards)

        Reads only ``1/num_shards`` of all examples in the data pipeline.

        :param shard_idx:
            The shard index.
        :param num_shards:
            The number of shards.

        :type shard_idx:
            int
        :type num_shards:
            int

        :rtype:
            DataPipelineBuilder

    .. method:: yield_from(fn)

        Applies ``fn`` to every example in the data pipeline and yields to the
        examples returned from the sub-data pipelines.

        :param fn:
            The function to apply.

        :type fn:
            ~typing.Callable[[~typing.Any], DataPipeline]

        :rtype:
            DataPipelineBuilder

    .. method:: and_return()

        Returns a new :class:`DataPipeline` instance.

        :rtype:
            DataPipeline
