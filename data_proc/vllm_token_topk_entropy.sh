#!/bin/bash

#SBATCH --output=/fsx-ram/lidli/slurm_logs/slurm-%A-%a.out
#SBATCH --error=/fsx-ram/lidli/slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=token_entropies
#SBATCH --nodes=1
#SBATCH --account=ram_external
#SBATCH --qos=ram_external_high
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --mem=0GB
#SBATCH --signal=USR1@90
#SBATCH --open-mode=append
#SBATCH --time=9:00:00
#SBATCH --array=0-15
#SBATCH --wckey=submitit

shard_id_num=$SLURM_ARRAY_TASK_ID
shard_id=$(printf '%04d' $shard_id_num)


# decoding params
seed=42
prompt_logprobs=200
max_logprobs=${prompt_logprobs}

# models to use
model_path='/fsx-ram/shared/Llama-3.2-1B'
tensor_parallel_size=8

data_dir="/fsx-chinchilla/thaottn/natural_reasoning_data_extracted"
output_dir="/fsx-ram/lidli/datasets/natural_reasoning_data_extracted_w_token_entropy"

input_path="${data_dir}/data.chunk.${shard_id}.jsonl"
output_path="${output_dir}/data.chunk.${shard_id}.jsonl"


if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

chunk_limit=1024

# max_model_len
if [ ! -d "$output_dir" ]; then
  mkdir -p $output_dir
fi
python3 -m ram.eval.vllm_inference \
	--prompt_path $input_path \
	--model $model_path \
	--output_path $output_path \
	--tensor_parallel_size ${tensor_parallel_size} \
	--inject_generation_config_json True \
	--max_tokens 1 --chunk_limit ${chunk_limit} --seed $seed \
	--max_logprobs ${max_logprobs} \
    --prompt_logprobs ${prompt_logprobs} --detokenize False \
	--entropy_in_output True --gpu_memory_utilization 0.3