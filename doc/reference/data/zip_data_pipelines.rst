read_sequence
=============

.. currentmodule: fairseq2

.. function:: data.data_pipeline._zip_data_pipelines(data_pipelines)

    Builds a data pipeline by zipping together ``data_pipelines``.

    :param data_pipelines:
        The data pipelines to zip.

    :type data_pipelines:
        ~typing.Sequence[DataPipeline]

    :rtype:
        DataPipelineBuilder
