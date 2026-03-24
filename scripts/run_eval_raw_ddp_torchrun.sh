#!/bin/bash
set -e

export MESA_GL_VERSION_OVERRIDE=4.1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_BLOCKING_WAIT=1

REPO_ROOT="/home/yewon/RoboVLMs"

ckpt_dir="$1"
config_path="$2"

GPUS_PER_NODE=1

cd "$REPO_ROOT"

echo "REPO_ROOT: $REPO_ROOT"
echo "ckpt_dir: $ckpt_dir"
echo "config_path: $config_path"
echo "cwd: $(pwd)"

#ls -l "$REPO_ROOT/eval/calvin"

torchrun --nnodes=1 --nproc_per_node=$GPUS_PER_NODE --master_port=6067 \
    "$REPO_ROOT/eval/calvin/evaluate_ddp-v2.py" \
    --config_path "$config_path" \
    --ckpt_path "$ckpt_dir" \
    --ckpt_idx 0 \
    --raw_calvin
'''
# Run
#conda activate robovlm

export MESA_GL_VERSION_OVERRIDE=4.1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_BLOCKING_WAIT=1
# export CUDA_VISIBLE_DEVICES=1

cd $EVALUTION_ROOT
ckpt_dir=$1
config_path=$2
sudo chmod 666 -R $ckpt_dir
GPUS_PER_NODE=8

torchrun --nnodes=1 --nproc_per_node=$GPUS_PER_NODE --master_port=6067 eval/calvin/evaluate_ddp-v2.py \
--config_path $config_path \
--ckpt_path $ckpt_dir \
--ckpt_idx 0 --raw_calvin
'''
