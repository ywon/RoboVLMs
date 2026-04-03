#!/usr/bin/env bash

PYTHON_BIN="/home/yewon/.conda/envs/robovlms/bin/python"
SCRIPT="/home/yewon/RoboVLMs/eval/calvin/eval_kosmos_finetune.py"

#CONFIG="/home/yewon/RoboVLMs/pretrained/robovlms/configs/kosmos_ph_calvin_abcd.json"
#CKPT="/home/yewon/RoboVLMs/pretrained/robovlms/checkpoints/kosmos_ph_calvin_abcd.pt"
CONFIG="/home/yewon/RoboVLMs/pretrained/robovlms/configs/kosmos_ph_oxe-pretrain.json"
CKPT="/home/yewon/RoboVLMs/pretrained/robovlms/checkpoints/kosmos_ph_oxe-pretrain.pt"
#CONFIG="/home/yewon/RoboVLMs/pretrained/robovlms/configs/kosmos_ph_finetune_eval.json"
#CKPT="/home/yewon/RoboVLMs/runs/checkpoints/kosmos/calvin_finetune/2026-03-31/kosmos_oxe2calvin_lora/last.ckpt"
DATASET="/mnt/hdd/calvin/calvin_ABCD_D/task_ABCD_D"
EVAL_JSON="/home/yewon/RoboVLMs/configs/data/calvin/eval_sequences_singlestep.json"

OUTPUT_DIR="/home/yewon/RoboVLMs/eval_results_oxe/4tasks_eval"
# torchrun 대신 단일 GPU에서 바로 실행
sudo CUDA_VISIBLE_DEVICES=0 \
"$PYTHON_BIN" "$SCRIPT" \
  --config_path "$CONFIG" \
  --ckpt_path "$CKPT" \
  --dataset_path "$DATASET" \
  --eval_json "$EVAL_JSON" \
  --output_dir "$OUTPUT_DIR" \
  --device_id 0 \
  --raw_calvin \
  --debug