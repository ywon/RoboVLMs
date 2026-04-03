#!/bin/bash
sudo env -u LD_LIBRARY_PATH \
PYTHONPATH=/home/yewon/RoboVLMs/calvin/calvin_models:/home/yewon/RoboVLMs/calvin:/home/yewon/RoboVLMs/LLaVA:/home/yewon/RoboVLMs:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 \
/home/yewon/.conda/envs/robovlms/bin/python \
robovlm_calvin_experiment/fine_tune.py \
pretrained/robovlms/configs/kosmos_ph_finetune.json --gpus 1