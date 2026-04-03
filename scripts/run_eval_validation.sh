#!/usr/bin/env bash

PYTHON_BIN="/home/yewon/.conda/envs/robovlms/bin/python"
SCRIPT="/home/yewon/RoboVLMs/eval/calvin/eval_kosmos_validation.py"

#CONFIG="/home/yewon/RoboVLMs/pretrained/robovlms/configs/kosmos_ph_finetune_eval.json"
#CONFIG="/home/yewon/RoboVLMs/pretrained/robovlms/configs/kosmos_ph_calvin_abcd.json"
CONFIG="/home/yewon/RoboVLMs/pretrained/robovlms/configs/kosmos_ph_oxe-pretrain.json"
#CKPT="/home/yewon/RoboVLMs/runs/checkpoints/kosmos/calvin_finetune/2026-03-31/kosmos_oxe2calvin_lora/last.ckpt"
#CKPT="/home/yewon/RoboVLMs/pretrained/robovlms/checkpoints/kosmos_ph_calvin_abcd.pt"
CKPT="/home/yewon/RoboVLMs/pretrained/robovlms/checkpoints/kosmos_ph_oxe-pretrain.pt"
DATASET="/mnt/hdd/calvin/calvin_ABCD_D/task_ABCD_D"
VAL_SEQ_JSON="/home/yewon/RoboVLMs/configs/data/calvin/eval_sequencesss.json"
OUTPUT_DIR="/home/yewon/RoboVLMs/eval_results_oxe/validation_eval"

sudo CUDA_VISIBLE_DEVICES=0 \
"$PYTHON_BIN" "$SCRIPT" \
  --config_path "$CONFIG" \
  --ckpt_path "$CKPT" \
  --dataset_path "$DATASET" \
  --validation_sequences_json "$VAL_SEQ_JSON" \
  --output_dir "$OUTPUT_DIR" \
  --device_id 0 \
  --debug \
  --raw_calvin \
  --save_initial_frame \
  --target_tasks \
    lift_blue_block_table \
    lift_pink_block_table \
    lift_red_block_table \
    place_in_drawer
