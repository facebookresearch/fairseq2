src_ckpt_dir="/fsx-ram/jacklanchantin/projects/ram_self_augmenting/experiments/llama3_1_1b_midtraining_data4_lr1e5_warm2000_cosdecay_8nodes_v6/checkpoints"
dest_ckpt_dir="/checkpoint/ram-external/lidli/checkpoints"
sub_dir="llama3_1_1b_midtraining_data4_lr1e5_warm2000_cosdecay_8nodes_v6"
steps=32000
s3_path="s3://fairusersglobal/tmp/lidli"
is_cross_cluster="true"

# run in src machine
if [ "$is_cross_cluster" = "true" ]; then
    aws s3 cp --recursive "${src_ckpt_dir}/step_${steps}/" "${s3_path}/${sub_dir}/checkpoints/step_${steps}/"
    aws s3 cp "${src_ckpt_dir}/model.yaml" ${s3_path}/${sub_dir}/checkpoints/model.yaml
fi

# run in dest machine
mkdir -p ${sub_dir}/checkpoints/step_${steps}/hg
rsync -av --exclude 'model.safetensors' /datasets/pretrained-llms/Llama-3.2-1B/* ${dest_ckpt_dir}/${sub_dir}/checkpoints/step_${steps}/hg

if [ "$is_cross_cluster" = "true" ]; then
    aws s3 cp --recursive ${s3_path}/${sub_dir}/checkpoints/step_${steps} ${dest_ckpt_dir}/${sub_dir}/checkpoints/step_${steps}
    aws s3 cp ${s3_path}/${sub_dir}/checkpoints/model.yaml ${dest_ckpt_dir}/${sub_dir}/checkpoints/model.yaml
else
    cp -r "${src_ckpt_dir}/step_${steps}/" ${dest_ckpt_dir}/${sub_dir}/checkpoints/step_${steps}
    cp "${src_ckpt_dir}/model.yaml" ${dest_ckpt_dir}/${sub_dir}/checkpoints/model.yaml
fi
