from fairseq2.context import RuntimeContext
from fairseq2.recipes.lm import OnlineFinetuneConfig
from fairseq2.recipes.lm._online_finetune._recipe import (
    VllmActorsSection,
    OnlineFinetuneDatasetSection,
)
from fairseq2.recipes.lm._online_finetune._grpo import (
    GrpoFinetuneConfig,
    GrpoLossConfig,
)

from fairseq2.recipes.lm._online_finetune._remote_model import (
    HFRayActorConfig,
    RemoteRayModelHandler,
    VllmRayActorConfig,
    VllmEngineArgs,
)
from dataclasses import replace

from fairseq2.recipes.lm._online_finetune._rewards import (
    RewardSection,
    RewardModelConfig,
    VLLMOutputReward,
    VLLMOutputRewardHandler,
)
from fairseq2.recipes.lm._online_finetune._common import OnlineCriterionSection
from fairseq2.optim import AdamWConfig

from fairseq2.recipes.config import (
    ActivationCheckpointingSection,
    CommonSection,
    DatasetSection,
    FSDPSection,
    GangSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    RegimeSection,
    TextTokenizerSection,
    TrainerSection,
    GradAccumulationSection,
)


def register_online_finetune_configs(context: RuntimeContext) -> None:
    """
    Here we keep some wildchat presets that are known to work well
    """

    registry = context.get_config_registry(OnlineFinetuneConfig)

    preset = registry.decorator

    @preset("llama3_1_instruct")
    def llama3_1_instruct() -> OnlineFinetuneConfig:
        config = OnlineFinetuneConfig()
        config.regime.validate_before_training = True

        return config

    @preset("qwen3_4b_wildchat_grpo")
    def qwen3_4b_wildchat_grpo() -> OnlineFinetuneConfig:

        model_config = ModelSection(name="qwen3_4b")
        tokenizer_config = TextTokenizerSection(name="qwen3_4b")

        dataset_config = OnlineFinetuneDatasetSection(
            name="wildchat",
            path="./wildchat_1k",
            train_split="train",
            valid_split="valid",
            batch_size=1,
            src_key="qwen3_src",
            extras={"keep_jsonl_keys": ["prompt_raw"]},
        )

        policy_vllm_config = VllmRayActorConfig(
            ray_actor_name="vllm_policy",
            backend="vllm",
            num_replicas=2,
            init_update_process_group=True,
            vllm_engine_args=VllmEngineArgs(
                model="./Qwen3-4B/",
                tokenizer="./Qwen3-4B/",
                tensor_parallel_size=2,
                enforce_eager=False,
                gpu_memory_utilization=0.7,
                enable_chunked_prefill=True,
            ),
            vllm_sampling_params={
                "n": 1,
                "temperature": 1.0,
                "max_tokens": 8196,
                "logprobs": 0,
            },
        )

        reference_vllm_config = VllmRayActorConfig(
            ray_actor_name="vllm_reference",
            backend="vllm",
            num_replicas=1,
            init_update_process_group=True,
            vllm_engine_args=VllmEngineArgs(
                model="./Qwen3-4B/",
                tokenizer="./Qwen3-4B/",
                tensor_parallel_size=2,
                enforce_eager=False,
                gpu_memory_utilization=0.7,
                enable_chunked_prefill=True,
            ),
            vllm_sampling_params={
                "n": 1,
                "temperature": 1.0,
                "max_tokens": 1,
                "prompt_logprobs": 0,
                "detokenize": False,
            },
        )

        reward_hf_config = HFRayActorConfig(
            ray_actor_name="hf_reward",
            num_replicas=2,
            backend="hf",
            pipeline_name="athene_reward_pipeline",
            tensor_parallel_size=1,
            init_update_process_group=False,
        )

        crit_config = OnlineCriterionSection(
            name="grpo",
            config=GrpoFinetuneConfig(
                loss_config=GrpoLossConfig(
                    beta=0.001,
                    group_size=8,
                    log_rollouts=True,
                    forward_group_size=4,
                    validation_vllm_sampling_params={
                        "n": 1,
                        "temperature": 0.6,
                        "top_p": 0.9,
                    },
                ),
                vllm_model_actor_name="vllm_policy",
                vllm_reference_model_actor_name="vllm_reference",
                vllm_reward_model_actor_name="hf_reward",
                reward=RewardSection(
                    name="athene_verifier",
                    config=RewardModelConfig(prompt_key="prompt_raw"),
                ),
            ),
        )

        vllm_config = VllmActorsSection(
            ray_cluster_ip_address="tobechanged_via_cli",
            ray_actors=[policy_vllm_config, reference_vllm_config, reward_hf_config],
        )

        regime_updates = {
            "num_data_epochs": 30,
            "validate_at_start": True,
            "validate_every_n_steps": 50,
            "checkpoint_every_n_steps": 100,
            "keep_best_n_checkpoints": 100,
            "save_model_only": False,  # set to False if willing to resume interrupted training
            "save_as_hugging_face": True,
        }

        config = OnlineFinetuneConfig()
        config.model = model_config
        config.dataset = dataset_config
        config.tokenizer = tokenizer_config
        config.trainer.max_grad_norm = 1.0
        config.vllm = vllm_config
        config.criterion = crit_config
        config.optimizer.config.lr = 1e-6
        config.regime = replace(config.regime, **regime_updates)

        return config
