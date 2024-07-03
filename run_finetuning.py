# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from argparse import ArgumentParser, Namespace
from datetime import timedelta
from pathlib import Path

from fairseq2 import setup_extensions
from fairseq2.logging import get_log_writer
from fairseq2.recipes.logging import setup_basic_logging, setup_logging
from fairseq2.recipes.utils.log import exception_logger, log_config
from fairseq2.recipes.wav2vec2.asr.train import (
    load_wav2vec2_asr_trainer,
    wav2vec2_asr_train_presets,
)
from fairseq2.utils.cluster import Cluster, ClusterConfig

log = get_log_writer(__name__)
USER = os.getenv("USER")


def train_wav2vec2_asr_model(config, output_dir: Path) -> None:
    """Run wav2vec2 ASR finetuning.

    :param config:
        The job configuration.
    :param output_dir:
        The output directory to store checkpoints and logs.
    """

    with exception_logger(log):
        setup_extensions()
        setup_basic_logging(debug=False)
        log_file = output_dir.expanduser().joinpath("logs/rank_{rank}.log").resolve()
        setup_logging(log_file, debug=False)
        log_config(config, log, output_dir.joinpath("config.yaml"))
        trainer = load_wav2vec2_asr_trainer(config, output_dir)
        trainer()


def main(args: Namespace) -> None:
    preset = args.preset

    if preset.split("_")[-1] == "perf":
        num_gpus = args.gpu
    elif preset.split("_")[0] == "base":
        num_gpus = 2
    else:
        num_gpus = 4

    if "FAIR_ENV_CLUSTER" in os.environ:
        partition = "nllb,ust,devaccel,learnaccel,mms,learnfair"
        root_dir = Path(f"/checkpoint/{USER}")
    else:
        partition = "mms_high,lowest"
        root_dir = Path(f"/fsx-mms/{USER}")

    cluster_config = ClusterConfig(
        cluster="slurm",
        parallelism=1,
        partition=partition,
        num_nodes=(num_gpus + 7) // 8,
        num_gpus_per_node=min(num_gpus, 8),
        cpus_per_task=10,
        log_dir=root_dir,  # Temporary, updated below.
        timeout=timedelta(minutes=4000),
    )

    if preset.split("_")[-1] == "perf":
        job_configs = []
        job_output_dirs = []
        for seed in (2, 3, 4):
            config = wav2vec2_asr_train_presets.get(preset)
            config.seed = args.seed
            output_dir = root_dir.joinpath(
                f"wav2vec2_asr_perf/fairseq2/{preset}_{num_gpus}gpu_seed{config.seed}"
            )
            job_configs.append(config)
            job_output_dirs.append(output_dir)

        cluster_config.log_dir = output_dir.joinpath("submitit")
        cluster = Cluster(cluster_config)
        cluster.run_job_array(train_wav2vec2_asr_model, job_configs, job_output_dirs)
    else:
        config = wav2vec2_asr_train_presets.get(preset)
        config.seed = args.seed
        output_dir = root_dir.joinpath(f"wav2vec2_asr/{preset}_seed{config.seed}")
        cluster_config.log_dir = output_dir.joinpath("submitit")
        cluster = Cluster(cluster_config)
        cluster.run_job(train_wav2vec2_asr_model, config, output_dir)


if __name__ == "__main__":
    parser = ArgumentParser(prog="finetune_mms")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--preset", type=str, default="base_10h")
    args = parser.parse_args()
    main(args)
